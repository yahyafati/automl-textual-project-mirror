"""A STARTER KIT SCRIPT for SS26 AutoML Exam --- Modality III: Text

You are not expected to follow this script or be constrained to it.

For a test run:
1) Download datasets (see, README) at chosen path
2) Run the script: 
```
python run.py \
    --data-path <path-to-downloaded-data> \
    --dataset amazon \
    --epochs 1
```

"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import yaml

from automl.core import TextAutoML
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
    YelpDataset,
)

logger = logging.getLogger(__name__)

FINAL_TEST_DATASET = "yelp"


def main_loop(
    dataset: str,
    output_path: Path,
    data_path: Path,
    seed: int,
    approach: str,
    val_size: float = 0.2,
    vocab_size: int = 10000,
    token_length: int = 128,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.0001,
    weight_decay: float = 0.01,
    ffnn_hidden: int = 128,
    lstm_emb_dim: int = 128,
    lstm_hidden_dim: int = 128,
    fraction_layers_to_finetune: float = 1.0,
    data_fraction: int = 1.0,
    load_path: Path = None,
) -> None:
    match dataset:
        case "ag_news":
            dataset_class = AGNewsDataset
        case "imdb":
            dataset_class = IMDBDataset
        case "amazon":
            dataset_class = AmazonReviewsDataset
        case "dbpedia":
            dataset_class = DBpediaDataset
        case "yelp":
            dataset_class = YelpDataset
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    logger.info("Fitting Text AutoML")

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your AutoML system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.

    # Get the dataset and create dataloaders
    data_path = Path(data_path) if isinstance(data_path, str) else data_path
    data_info = dataset_class(data_path).create_dataloaders(
        val_size=val_size, random_state=seed
    )
    train_df = data_info["train_df"]

    _subsample = np.random.choice(
        list(range(len(train_df))),
        size=int(data_fraction * len(train_df)),
        replace=False,
    )
    train_df = train_df.iloc[_subsample]

    val_df = data_info.get("val_df", None)
    test_df = data_info["test_df"]
    num_classes = data_info["num_classes"]
    logger.info(
        f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}"
    )
    logger.info(f"Number of classes: {num_classes}")

    # Initialize the TextAutoML instance with the best parameters
    automl = TextAutoML(
        seed=seed,
        approach=approach,
        vocab_size=vocab_size,
        token_length=token_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        ffnn_hidden=ffnn_hidden,
        lstm_emb_dim=lstm_emb_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        fraction_layers_to_finetune=fraction_layers_to_finetune,
    )

    # Fit the AutoML model on the training and validation datasets
    val_err = automl.fit(
        train_df,
        val_df,
        num_classes=num_classes,
        load_path=load_path,
        save_path=output_path,
    )
    logger.info("Training complete")

    # Predict on the test set
    test_preds, test_labels = automl.predict(test_df)

    # Write the predictions of X_test to disk
    logger.info("Writing predictions to disk")
    with (output_path / "score.yaml").open("w") as f:
        yaml.safe_dump({"val_err": float(val_err)}, f)
    logger.info(f"Saved validation score at {output_path / 'score.yaml'}")
    with (output_path / "test_preds.npy").open("wb") as f:
        np.save(f, test_preds)
    logger.info(f"Saved the prediction at {output_path / 'test_preds.npy'}")

    # In case of running on the final exam data, also add the predictions.npy
    # to the correct location for auto evaluation.
    if dataset == FINAL_TEST_DATASET:
        test_output_path = output_path / "predictions.npy"
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # Check if test_labels has missing data
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"Accuracy on test set: {acc}")
        with (output_path / "score.yaml").open("a+") as f:
            yaml.safe_dump({"test_err": float(1 - acc)}, f)

        # Log detailed classification report for better insight
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(test_labels, test_preds)}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        logger.info(f"No test labels available for dataset '{dataset}'")

    return val_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to run on.",
        choices=["ag_news", "imdb", "amazon", "dbpedia", "yelp"],
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "The path to save the predictions to."
            " By default this will just save to the cwd as `./results`."
        ),
    )
    parser.add_argument(
        "--load-path",
        type=Path,
        default=None,
        help="The path to resume checkpoint from.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "The path to laod the data from."
            " By default this will look up cwd for `./.data/`."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using any randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        ),
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="transformer",
        choices=["tfidf", "lstm", "transformer"],
        help=(
            "The approach to use for the AutoML system. "
            "Options are 'tfidf', 'lstm', or 'transformer'."
        ),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="The size of the vocabulary to use for the text dataset.",
    )
    parser.add_argument(
        "--token-length",
        type=int,
        default=128,
        help="The maximum length of tokens to use for the text dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to use for training and evaluation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The learning rate to use for the optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="The weight decay to use for the optimizer.",
    )

    parser.add_argument(
        "--lstm-emb-dim",
        type=int,
        default=64,
        help="The embedding dimension to use for the LSTM model.",
    )

    parser.add_argument(
        "--lstm-hidden-dim",
        type=int,
        default=64,
        help="The hidden size to use for the LSTM model.",
    )

    parser.add_argument(
        "--ffnn-hidden-layer-dim",
        type=int,
        default=64,
        help="The hidden size to use for the model.",
    )

    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1,
        help="Subsampling of training set, in fraction (0, 1].",
    )
    args = parser.parse_args()

    logger.info(f"Running text dataset {args.dataset}\n{args}")

    if args.output_path is None:
        args.output_path = (
            Path.cwd().absolute()
            / "results"
            / f"dataset={args.dataset}"
            / f"seed={args.seed}"
        )
    if args.data_path is None:
        args.data_path = Path.cwd().absolute() / ".data"

    args.output_path = Path(args.output_path).absolute()
    args.output_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, filename=args.output_path / "run.log")

    main_loop(
        dataset=args.dataset,
        output_path=Path(args.output_path).absolute(),
        data_path=Path(args.data_path).absolute(),
        seed=args.seed,
        approach=args.approach,
        vocab_size=args.vocab_size,
        token_length=args.token_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ffnn_hidden=args.ffnn_hidden_layer_dim,
        lstm_emb_dim=args.lstm_emb_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        data_fraction=args.data_fraction,
        load_path=Path(args.load_path) if args.load_path is not None else None,
    )
# end of file
