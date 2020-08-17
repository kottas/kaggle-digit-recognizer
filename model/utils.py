import numpy as np
import pandas as pd
import torch

from torch.utils.data.dataset import Dataset
from typing import Dict, List, Optional, Union


class MNISTDataset(Dataset):
    """ MNIST Dataset in PyTorch """

    def __init__(self, features: List[np.ndarray], labels: List[int]):
        """
        Initialization

        Parameters
        ----------
        features: List[np.ndarray]
           Image features
        labels: List[int]
           Image labels
        """
        self.dataset = pd.DataFrame(dict(labels=labels, features=features))

    def __len__(self) -> int:
        """ Length """
        return len(self.dataset)

    def __getitem__(self, index) -> Dict[str, Union[int, np.ndarray]]:
        """ Get item """
        return self.dataset.iloc[index].to_dict()

    @classmethod
    def from_file(cls, path: str) -> "MNISTDataset":
        """
        Initialize dataset from file

        Parameters
        ----------
        path: str
           Path to file

        Returns
        -------
        dataset: MNISTDataset
           Initialized instance
        """
        dataset = pd.read_csv(path)
        columns = dataset.columns
        if "label" in dataset.columns:
            labels = dataset["label"].tolist()
            columns = dataset.columns[1:]
        else:
            labels = [-1] * len(dataset)
        features = list(dataset[columns].to_numpy().reshape((len(dataset), 28, 28)) / 255)

        return cls(features=features, labels=labels)

    @classmethod
    def split(
        cls, dataset: "MNISTDataset", ratio: float = 0.2, seed: Optional[int] = None
    ) -> Dict[str, "MNISTDataset"]:
        """
        Dataset split-per-label, to training and validation sets

        Parameters
        ----------
        dataset: MNISTDataset
           Dataset to split
        ratio: float
           Split ratio
        seed: Optional[int]
           Random seed

        Returns
        -------
        split: Dict[str, "MNISTDatasetPT"]
           Train-val datasets. {"train": train, "val": val}
        """
        data = dataset.dataset
        val_set = pd.concat(
            [
                data[data["labels"] == label].sample(frac=ratio, random_state=seed)
                for label in data["labels"].unique()
            ]
        )
        train_set = data[~data.index.isin(val_set.index)]

        # initialize
        train = cls(features=train_set["features"].tolist(), labels=train_set["labels"].tolist())
        val = cls(features=val_set["features"].tolist(), labels=val_set["labels"].tolist())

        return dict(train=train, val=val)


def collate(batch: List[Dict[str, Union[int, np.ndarray]]]) -> Dict[str, torch.Tensor]:
    """ PyTorch collate function """
    return dict(
        features=torch.tensor([item["features"] for item in batch], dtype=torch.float32).unsqueeze(
            dim=1
        ),
        labels=torch.tensor([item["labels"] for item in batch]),
    )
