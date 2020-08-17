import argparse
import json
import os
import subprocess

from typing import Optional

from model.classifier import MNISTClassifier
from model.utils import MNISTDataset


MAIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "data")
KAGGLE_JSON = os.path.join(DATA_DIR, "kaggle.json")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv.zip")
TEST_FILE = os.path.join(DATA_DIR, "test.csv.zip")
CONFIG_FILE = os.path.join(os.path.join(MAIN_DIR, "scripts"), "config.json")
OUTPUT_DIR = os.path.join(MAIN_DIR, "results")


def argument_parser():
    """ Command line argument parser """
    parser = argparse.ArgumentParser(description="MNIST digit recognizer")
    parser.add_argument("-config", type=str, default=CONFIG_FILE, help="Configuration file")
    parser.add_argument("-dir_in", type=str, default=DATA_DIR, help="Data folder")
    parser.add_argument("-dir_out", type=str, default=OUTPUT_DIR, help="Output dir")
    parser.add_argument("-kaggle_json", type=str, default=None, help="Kaggle JSON token")
    parser.add_argument("-ratio", type=float, default=None, help="Train-val split ratio")

    return parser.parse_args()


def main(
    dir_in: str = DATA_DIR,
    fname_config: str = CONFIG_FILE,
    dir_out: str = OUTPUT_DIR,
    kaggle_json: Optional[str] = None,
    split_ratio: Optional[float] = None,
):
    """
    Kaggle MNIST classifier

    Parameters
    ----------
    dir_in: str
       Data folder
    fname_config: str
       Configuration file
    dir_out: str
       Output folder
    kaggle_json: Optional[str]
       Kaggle JSON token
    split_ratio: Optional[float]
       Train-val split ratio
    """
    print(f"Configuration: {fname_config}")
    with open(fname_config, "r") as f:
        configuration = json.loads(f.read())

    # load datasets
    paths = {"train": os.path.join(dir_in, "train.csv"), "test": os.path.join(dir_in, "test.csv")}
    if not all([os.path.isfile(path) for _, path in paths.items()]):
        download_datasets(kaggle_token=kaggle_json, dir_in=dir_in)
    train_set = MNISTDataset.from_file(path=paths["train"])
    test_set = MNISTDataset.from_file(path=paths["test"])

    # split
    val_set: Optional[MNISTDataset] = None
    if split_ratio is not None:
        print(f"Splitting ratio: {split_ratio}")
        seed = None if configuration["deterministic"] else 1
        datasets = MNISTDataset.split(dataset=train_set, ratio=split_ratio, seed=seed)
        train_set = datasets["train"]
        val_set = datasets["val"]

    # train
    model = MNISTClassifier(**configuration)
    model.train(train_set=train_set, val_set=val_set)

    # predict
    predictions = model.predict(dataset=test_set)

    # save
    os.makedirs(dir_out, exist_ok=True)
    predictions.index += 1
    predictions.to_csv(
        os.path.join(dir_out, "predictions.csv"), columns=["Label"], index_label="ImageId"
    )


def download_datasets(kaggle_token: str, dir_in: str):
    """
    Download Kaggle datasets

    Parameters
    ----------
    kaggle_token: str
       Path to Kaggle token JSON file
       See: https://github.com/Kaggle/kaggle-api#api-credentials
    dir_in: str
       Download folder
    """
    kaggle_dir, _ = os.path.split(kaggle_token)
    os.environ["KAGGLE_CONFIG_DIR"] = kaggle_dir
    os.makedirs(dir_in, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", "digit-recognizer", "-p", dir_in])
    subprocess.run(["unzip", f"{os.path.join(dir_in, 'digit-recognizer.zip')}", "-d", dir_in])


if __name__ == "__main__":
    args = argument_parser()
    main(
        dir_in=args.dir_in,
        fname_config=args.config,
        dir_out=args.dir_out,
        kaggle_json=args.kaggle_json,
        split_ratio=args.ratio,
    )
