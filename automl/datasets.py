"""Dataset classes for NLP AutoML tasks."""
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import string
from sklearn.model_selection import train_test_split


class BaseTextDataset(ABC):
    """Base class for text datasets."""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.vocab_size = 10000  # Default vocab size
        self.max_length = 512    # Default max sequence length
        
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data."""
        pass
    
    @abstractmethod
    def get_num_classes(self) -> int:
        """Return number of classes."""
        pass
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def create_dataloaders(
        self,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Create train/validation/test dataloaders and preprocessing objects."""
        train_df, test_df = self.load_data()  # not implemented in base class `BaseTextDataset`
        
        # Split training data into train/validation
        if val_size > 0:
            train_df, val_df = train_test_split(
                train_df, test_size=val_size, random_state=random_state,
                stratify=train_df['label'] if 'label' in train_df.columns else None
            )
        else:
            val_df = None
        
        # Preprocess text
        train_df['text'] = train_df['text'].apply(self.preprocess_text)
        if val_df is not None:
            val_df['text'] = val_df['text'].apply(self.preprocess_text)
        test_df['text'] = test_df['text'].apply(self.preprocess_text)
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'num_classes': self.get_num_classes()
        }


class AGNewsDataset(BaseTextDataset):
    """AG News dataset for news categorization (4 classes)."""
    
    def get_num_classes(self) -> int:
        return 4
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load AG News data."""
        # This assumes CSV files with columns: label, text
        train_path = self.data_path / "ag_news" / "train.csv"
        test_path = self.data_path / "ag_news" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            raise FileNotFoundError(f"Data files not found at {train_path}, generating dummy data...")
        
        return train_df, test_df


class IMDBDataset(BaseTextDataset):
    """IMDB movie review sentiment dataset (2 classes)."""
    
    def get_num_classes(self) -> int:
        return 2
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load IMDB data."""
        train_path = self.data_path / "imdb" / "train.csv"
        test_path = self.data_path / "imdb" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            raise FileNotFoundError(f"Data files not found at {train_path}, generating dummy data...")
        
        return train_df, test_df


class AmazonReviewsDataset(BaseTextDataset):
    """Amazon product reviews dataset (5 classes for categories)."""
    
    def get_num_classes(self) -> int:
        return 5
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Amazon reviews data."""
        train_path = self.data_path / "amazon" / "train.csv"
        test_path = self.data_path / "amazon" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            raise FileNotFoundError(f"Data files not found at {train_path}, generating dummy data...")
        
        return train_df, test_df


class DBpediaDataset(BaseTextDataset):
    """DBpedia ontology classification dataset (14 classes)."""
    
    def get_num_classes(self) -> int:
        return 14
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load DBpedia ontology data."""
        train_path = self.data_path / "dbpedia" / "train.csv"
        test_path = self.data_path / "dbpedia" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            raise FileNotFoundError(f"Data files not found at {train_path}, generating dummy data...")

        # Crucial handling of negative class label
        train_df['label'] = train_df['label'].replace(-1, self.get_num_classes() - 1)
        test_df['label'] = test_df['label'].replace(-1, self.get_num_classes() - 1)

        return train_df, test_df


class YelpDataset(BaseTextDataset):
    """Yelp Reviews 5-star rating dataset."""
    
    def get_num_classes(self) -> int:
        return 5
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Yelp 5-star data."""
        train_path = self.data_path / "yelp" / "train.csv"
        test_path = self.data_path / "yelp" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            raise FileNotFoundError(f"Data files not found at {train_path}, generating dummy data...")
        
        return train_df, test_df
