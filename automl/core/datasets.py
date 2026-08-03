"""Dataset classes for NLP AutoML tasks."""
import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, TypedDict

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplits(TypedDict):
    train_df: pd.DataFrame
    val_df: Optional[pd.DataFrame]
    test_df: pd.DataFrame
    num_classes: int


class BaseTextDataset(ABC):
    """Base class for text datasets."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.vocab_size = 10000  # Default vocab size
        self.max_length = 512  # Default max sequence length

        # Cache for load_data() results so repeated calls don't reload from disk
        self._cached_train_df: Optional[pd.DataFrame] = None
        self._cached_test_df: Optional[pd.DataFrame] = None

    @abstractmethod
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Actually load train and test data (subclasses implement this)."""
        pass

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data, caching the result in memory after first call."""
        if self._cached_train_df is None or self._cached_test_df is None:
            train_df, test_df = self._load_data()
            self._cached_train_df = train_df
            self._cached_test_df = test_df

        # Return copies so downstream mutation (e.g. preprocessing) doesn't
        # corrupt the cached originals.
        return self._cached_train_df.copy(), self._cached_test_df.copy()

    def clear_cache(self) -> None:
        """Clear the in-memory cache, forcing the next load_data() call to reload."""
        self._cached_train_df = None
        self._cached_test_df = None

    def _read_csv_pair(self, subdir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read train.csv/test.csv from data_path/subdir, assumed columns: label, text."""
        train_path = self.data_path / subdir / "train.csv"
        test_path = self.data_path / subdir / "test.csv"

        if not (train_path.exists() and test_path.exists()):
            raise FileNotFoundError(f"Data files not found at {train_path}")

        return pd.read_csv(train_path), pd.read_csv(test_path)

    @abstractmethod
    def get_num_classes(self) -> int:
        """Return number of classes."""
        pass

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Basic text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    @staticmethod
    def _uniform_sample(
        df: pd.DataFrame,
        max_num_rows: int,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Uniformly (stratified by label, if present) subsample df down to
        at most max_num_rows rows."""
        if max_num_rows is None or len(df) <= max_num_rows:
            return df

        if "label" not in df.columns:
            return df.sample(n=max_num_rows, random_state=random_state).sort_index()

        n_classes = df["label"].nunique()
        base_n_per_class = max_num_rows // n_classes
        remainder = max_num_rows - base_n_per_class * n_classes

        # Give the remainder to classes in a deterministic order so results
        # are reproducible.
        class_labels = sorted(df["label"].unique(), key=str)
        extra_alloc = {
            lbl: (1 if i < remainder else 0) for i, lbl in enumerate(class_labels)
        }

        sampled_parts = []
        for lbl in class_labels:
            group = df[df["label"] == lbl]
            n = base_n_per_class + extra_alloc[lbl]
            n = min(n, len(group))  # can't sample more than the group has
            sampled_parts.append(
                group.sample(n=n, random_state=random_state)
                if n > 0
                else group.iloc[0:0]
            )

        sampled_df = pd.concat(sampled_parts).sort_index()

        # If some classes were too small to fill their quota, top up from
        # the remaining pool to still hit max_num_rows (best effort).
        shortfall = max_num_rows - len(sampled_df)
        if shortfall > 0:
            leftover = df.drop(sampled_df.index)
            if leftover is not None and len(leftover) > 0:
                top_up = leftover.sample(
                    n=min(shortfall, len(leftover)), random_state=random_state
                )
                sampled_df = pd.concat([sampled_df, top_up]).sort_index()

        return sampled_df

    def create_dataloaders(
        self,
        val_size: float = 0.2,
        random_state: int = 42,
        train_fraction: float = 1.0,
        max_num_rows: Optional[int] = None,
    ) -> DatasetSplits:
        """Create train/validation/test dataloaders and preprocessing objects.

        Args:
            val_size: Fraction of training data to hold out for validation.
            random_state: Seed for reproducibility.
            train_fraction: Fraction of training data to keep (stratified by label).
            max_num_rows: If set, caps the total number of training rows
                (applied after train_fraction), sampled uniformly across
                classes when a "label" column is present.
        """

        train_df, test_df = self.load_data()

        # --- apply train_fraction ---
        if train_fraction < 1.0:
            sampled_indices = (
                train_df.groupby("label", group_keys=False)
                .sample(frac=train_fraction, random_state=random_state)
                .index
            )

            train_df = train_df.loc[sampled_indices].sort_index()

        # --- apply max_num_rows (uniform/stratified cap) ---
        if max_num_rows is not None:
            train_df = self._uniform_sample(
                train_df, max_num_rows=max_num_rows, random_state=random_state
            )

        # Split training data into train/validation
        val_df: Optional[pd.DataFrame] = None
        if val_size > 0:
            train_df, val_df = train_test_split(
                train_df,
                test_size=val_size,
                random_state=random_state,
                stratify=train_df["label"] if "label" in train_df.columns else None,
            )

        # Preprocess text
        train_df["text"] = train_df["text"].apply(self.preprocess_text)
        if val_df is not None:
            val_df["text"] = val_df["text"].apply(self.preprocess_text)
        test_df["text"] = test_df["text"].apply(self.preprocess_text)

        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "num_classes": self.get_num_classes(),
        }


class AGNewsDataset(BaseTextDataset):
    """AG News dataset for news categorization (4 classes)."""

    def get_num_classes(self) -> int:
        return 4

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._read_csv_pair("ag_news")


class IMDBDataset(BaseTextDataset):
    """IMDB movie review sentiment dataset (2 classes)."""

    def get_num_classes(self) -> int:
        return 2

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._read_csv_pair("imdb")


class AmazonReviewsDataset(BaseTextDataset):
    """Amazon product reviews dataset (5 classes for categories)."""

    def get_num_classes(self) -> int:
        return 5

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._read_csv_pair("amazon")


class DBpediaDataset(BaseTextDataset):
    """DBpedia ontology classification dataset (14 classes)."""

    def get_num_classes(self) -> int:
        return 14

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = self._read_csv_pair("dbpedia")

        # Crucial handling of negative class label
        n = self.get_num_classes() - 1
        train_df["label"] = train_df["label"].replace(-1, n)
        test_df["label"] = test_df["label"].replace(-1, n)

        return train_df, test_df


class YelpDataset(BaseTextDataset):
    """Yelp Reviews 5-star rating dataset."""

    def get_num_classes(self) -> int:
        return 5

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._read_csv_pair("yelp")


def get_dataset_class(dataset: str):
    match dataset:
        case "ag_news":
            return AGNewsDataset
        case "imdb":
            return IMDBDataset
        case "amazon":
            return AmazonReviewsDataset
        case "dbpedia":
            return DBpediaDataset
        case "yelp":
            return YelpDataset
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")
