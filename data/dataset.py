import csv
import random

import fire
import kagglehub

import constants


def get_data():
    # Download latest version
    print("Downloading dataset")
    path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")

    print("Path to dataset files:", path)


if __name__ == "__main__":
    fire.Fire(get_data)


class PositionDataPoint:
    def __init__(self, fen, eval, best_move):
        self.fen = fen
        self.best_move = best_move
        self.eval = eval


def get_splits(
    path: str,
) -> tuple[list[PositionDataPoint], list[PositionDataPoint], list[PositionDataPoint]]:
    """Gets train, val, and test splits

    Args:
        path (str): path to csv

    Returns:
        tuple[list[PositionDataPoint], list[PositionDataPoint], list[PositionDataPoint]]: train, val, and test split in that order
    """
    val_size = 500
    test_size = 500
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            data.append(PositionDataPoint(row[0], row[1], row[2]))

    if val_size + test_size > len(data):
        raise ValueError

    random.seed(constants.SEED)
    random.shuffle(data)

    return (
        data[val_size + test_size :],
        data[:val_size],
        data[val_size : val_size + test_size],
    )
