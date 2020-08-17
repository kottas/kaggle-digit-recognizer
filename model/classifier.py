import argparse
import pandas as pd
import pytorch_lightning as pl
import torch

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Union

from .utils import MNISTDataset, collate


class MNISTClassifier:
    """ MNIST classifier """

    def __init__(
        self,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        auto_lr_find: bool = True,
        early_stopping: bool = True,
        enable_logger: bool = True,
        deterministic: bool = False,
        half_precision: bool = False,
    ):
        """
        Initialize

        Parameters
        ----------
        epochs: int
           Training epochs
        learning_rate: float
           Learning rate
        batch_size: int
           Batch size
        auto_lr_find: bool
           Automatic learning rate finder
        early_stopping: bool
           Early stopping
        enable_logger: bool
           Enable logger
        deterministic: bool
           Deterministic
        half_precision: bool
           Half precision training
        """
        # attributes
        self.epochs = epochs
        self.batch_size = batch_size
        self.auto_lr_find = auto_lr_find
        self.early_stopping = early_stopping
        self.enable_logger = enable_logger
        self.deterministic = deterministic
        self.half_precision = half_precision

        # model
        self.model = MNISTModel(hparams=argparse.Namespace(learning_rate=learning_rate))

    def train(self, train_set: MNISTDataset, val_set: Optional[MNISTDataset] = None):
        """
        Train model

        Parameters
        ----------
        train_set: MNISTDataset
           Training dataset
        val_set: Optional[MNISTDataset]
           Validation dataset
        """
        self.early_stopping = False if val_set is None else self.early_stopping
        val_set = (
            val_set if val_set is not None else MNISTDataset(
                features=train_set.dataset["features"][:self.batch_size].tolist(),
                labels=train_set.dataset["labels"][:self.batch_size].tolist()
            )
        )

        # initialize dataloader
        train_dataloader = DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=4,
        )
        val_dataloader = DataLoader(
            dataset=val_set,
            batch_size=self.batch_size * 2,
            shuffle=False,
            collate_fn=collate,
            num_workers=4,
        )

        # trainer
        trainer = pl.Trainer(
            logger=self.enable_logger,
            checkpoint_callback=False,
            early_stop_callback=pl.callbacks.EarlyStopping() if self.early_stopping else False,
            gradient_clip_val=1.0,
            max_epochs=self.epochs,
            auto_lr_find=self.auto_lr_find,
            gpus=torch.cuda.device_count(),
            progress_bar_refresh_rate=1 if self.enable_logger else 0,
            precision=32 if not self.half_precision else 16,
            weights_summary=None,
            deterministic=self.deterministic,
        )

        # fit
        trainer.fit(
            model=self.model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
        )

    def predict(self, dataset: MNISTDataset) -> pd.DataFrame:
        """
        Predict

        Parameters
        ----------
        dataset: MNISTDataset
           Dataset

        Returns
        -------
        results: pd.DataFrame
           Results. ["label", "probability"]
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.freeze()

        # dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            collate_fn=collate,
            num_workers=4,
        )

        # predict
        predictions, probabilities = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                _, probs = self.model(batch["features"].to(device))
                max_probs, preds = probs.topk(k=1)
                predictions.extend(preds.flatten().tolist())
                probabilities.extend(max_probs.flatten().tolist())

        return pd.DataFrame(dict(Label=predictions, Probability=probabilities))


class MNISTModel(pl.LightningModule):
    """ Neural Network for MNIST classification """

    def __init__(self, hparams: argparse.Namespace):
        """
        Initialization

        Parameters
        ----------
        hparams: Namespace
           Hyperparameters.
        """
        super().__init__()
        self.learning_rate = hparams.learning_rate

        # model
        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.4),
        )
        self.cnn2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.4),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=64 * int((28 / 4) ** 2), out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
        )
        self.dense2 = torch.nn.Linear(in_features=128, out_features=10)

        # activation
        self.softmax = torch.nn.Softmax()

        # loss
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        """ Forward pass """
        out1 = self.cnn1(inputs)
        out2 = self.cnn2(out1)
        out3 = self.dense1(out2)
        logits = self.dense2(out3)
        probabilities = self.softmax(logits)

        return logits, probabilities

    def configure_optimizers(self) -> Optimizer:
        """ Configure optimizer """
        return torch.optim.AdamW(params=self.parameters(), lr=self.learning_rate)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """ Training step """
        # forward pass
        logits, _ = self(batch["features"])

        # logs
        loss = self.cross_entropy_loss(logits, batch["labels"])

        return dict(loss=loss)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """ Validation step """
        # forward pass
        logits, probabilities = self(batch["features"])

        # loss
        loss = self.cross_entropy_loss(logits, batch["labels"])

        # predictions
        predictions = torch.argmax(probabilities, dim=1)

        return dict(predictions=predictions, labels=batch["labels"], loss=loss)

    def validation_epoch_end(
        self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, torch.Tensor]:
        """ Epoch calculation for validation set """
        # accuracy
        predictions = torch.cat([output["predictions"] for output in outputs])
        labels = torch.cat([output["labels"] for output in outputs])
        val_accuracy = (predictions == labels).sum().to(dtype=torch.float) / len(labels)

        # mean loss
        val_loss = torch.stack([output["loss"] for output in outputs]).mean()

        return dict(
            val_loss=val_loss,
            progress_bar=dict(val_loss=val_loss, val_accuracy_=val_accuracy),
            log=dict(val_loss=val_loss, val_accuracy=val_accuracy),
        )
