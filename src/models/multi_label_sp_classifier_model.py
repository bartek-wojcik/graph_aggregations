from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
from src.models.modules import gat, jumping_knowledge, graph_sage, gcn


class MultiLabelSuperpixelClassifierModel(LightningModule):
    """LightningModule for image classification from superpixels."""

    def __init__(
        self,
        architecture: str = "GraphSAGE",
        aggregation_method: str = "concat",
        num_node_features: int = 3,
        num_conv_layers: int = 3,
        conv_size: int = 128,
        lin_size: int = 128,
        output_size: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        # init network architecture
        if self.hparams.architecture == "GraphSAGE":
            self.model = graph_sage.GraphSage(hparams=self.hparams)
        elif self.hparams.architecture == "GAT":
            self.model = gat.GAT(hparams=self.hparams)
        elif self.hparams.architecture == "JumpingKnowledge":
            self.model = jumping_knowledge.JK(hparams=self.hparams)
        elif self.hparams.architecture == "GCN":
            self.model = gcn.GCN(hparams=self.hparams)
        else:
            raise Exception("Incorrect architecture name!")

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def forward(self, batch: Any):
        return self.model(batch)

    def step(self, batch: Any):
        logits = self.forward(batch)
        loss = self.criterion(logits, batch.y.type(torch.float))
        preds = logits > 0.5
        return loss, preds, batch.y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )