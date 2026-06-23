import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from automl.models import SimpleFFNN, LSTMClassifier
from automl.utils import SimpleTextDataset
from pathlib import Path
import logging
from typing import Tuple
from collections import Counter

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextAutoML:
    def __init__(
        self,
        seed=42,
        approach="auto",
        vocab_size=10000,
        token_length=128,
        epochs=5,
        batch_size=64,
        lr=1e-4,
        weight_decay=0.0,
        ffnn_hidden=128,
        lstm_emb_dim=128,
        lstm_hidden_dim=128,
        fraction_layers_to_finetune: float=1.0,
    ):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.approach = approach
        self.vocab_size = vocab_size
        self.token_length = token_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.ffnn_hidden = ffnn_hidden
        self.lstm_emb_dim = lstm_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fraction_layers_to_finetune = fraction_layers_to_finetune

        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.num_classes = None
        self.train_texts = []
        self.train_labels = []
        self.val_texts = []
        self.val_labels = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_classes: int,
        approach=None,
        vocab_size=None,
        token_length=None,
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        ffnn_hidden=None,
        lstm_emb_dim=None,
        lstm_hidden_dim=None,
        fraction_layers_to_finetune=None,
        load_path: Path=None,
        save_path: Path=None,
    ):
        """
        Fits a model to the given dataset.

        Parameters:
        - train_df (pd.DataFrame): Training data with 'text' and 'label' columns.
        - val_df (pd.DataFrame): Validation data with 'text' and 'label' columns.
        - num_classes (int): Number of classes in the dataset.
        - seed (int): Random seed for reproducibility.
        - approach (str): Model type - 'tfidf', 'lstm', or 'transformer'. Default is 'auto'.
        - vocab_size (int): Maximum vocabulary size.
        - token_length (int): Maximum token sequence length.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay for optimizer.
        - ffnn_hidden (int): Hidden dimension size for FFNN.
        - lstm_emb_dim (int): Embedding dimension size for LSTM.
        - lstm_hidden_dim (int): Hidden dimension size for LSTM.
        """
        if approach is not None: self.approach = approach
        if vocab_size is not None: self.vocab_size = vocab_size
        if token_length is not None: self.token_length = token_length
        if epochs is not None: self.epochs = epochs
        if batch_size is not None: self.batch_size = batch_size
        if lr is not None: self.lr = lr
        if weight_decay is not None: self.weight_decay = weight_decay
        if ffnn_hidden is not None: self.ffnn_hidden = ffnn_hidden
        if lstm_emb_dim is not None: self.lstm_emb_dim = lstm_emb_dim
        if lstm_hidden_dim is not None: self.lstm_hidden_dim = lstm_hidden_dim
        if fraction_layers_to_finetune is not None: self.fraction_layers_to_finetune = fraction_layers_to_finetune
        
        logger.info("Loading and preparing data...")

        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        self.num_classes = num_classes
        logger.info(f"Train class distribution: {Counter(self.train_labels)}")
        logger.info(f"Val class distribution: {Counter(self.val_labels)}")

        dataset = None
        if self.approach == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.vocab_size,
                lowercase=True,
                min_df=2,    # ignore words appearing in less than 2 sentences
                max_df=0.8,  # ignore words appearing in > 80% of sentences
                sublinear_tf=True,  # use log-spaced term-frequency scoring
            )
            X = self.vectorizer.fit_transform(self.train_texts).toarray()
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(self.train_labels)
            )
            # models and dataloaders
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            X = self.vectorizer.transform(self.val_texts).toarray()
            _dataset = torch.utils.data.TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(self.val_labels)
            )
            val_loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)
            self.model = SimpleFFNN(
                X.shape[1], hidden=self.ffnn_hidden, output_dim=self.num_classes
            )

        elif self.approach in ['lstm', 'transformer']:
            model_name = 'distilbert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.vocab_size = self.tokenizer.vocab_size
            dataset = SimpleTextDataset(
                self.train_texts, self.train_labels, self.tokenizer, self.token_length
            )
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            _dataset = SimpleTextDataset(
                self.val_texts, self.val_labels, self.tokenizer, self.token_length
            )
            val_loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)

            match self.approach:
                case "lstm":
                    self.model = LSTMClassifier(
                        len(self.tokenizer),
                        self.lstm_emb_dim,
                        self.lstm_hidden_dim,
                        self.num_classes
                    )
                case "transformer":
                    if TRANSFORMERS_AVAILABLE:
                        self.model = AutoModelForSequenceClassification.from_pretrained(
                            model_name, 
                            num_labels=self.num_classes
                        )
                        freeze_layers(self.model, self.fraction_layers_to_finetune)  
                    else:
                        raise ValueError(
                            "Need `AutoTokenizer`, `AutoModelForSequenceClassification` "
                            "from `transformers` package."
                        )
                case _:
                    raise ValueError("Unsupported approach or missing transformers.")
        else:
            raise ValueError(f"Unrecognized approach: {self.approach}")
        
        # Training and validating
        self.model.to(self.device)
        assert dataset is not None, f"`dataset` cannot be None here!"
        val_acc = self._train_loop(
            train_loader,
            val_loader,
            load_path=load_path,
            save_path=save_path,
        )

        return 1 - val_acc

    def _train_loop(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        load_path: Path=None,
        save_path: Path=None,
    ):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        # handling checkpoint resume
        if load_path is not None:
            _states = torch.load(load_path / "checkpoint.pth", map_location='cpu')  # Load to CPU first
            self.model.load_state_dict(_states["model_state_dict"])
            optimizer.load_state_dict(_states["optimizer_state_dict"])
            start_epoch = _states["epoch"]
            logger.info(f"Resuming from checkpoint at {start_epoch}")

        val_acc = 0.0
        for epoch in range(start_epoch, self.epochs):            
            total_loss = 0
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()

                if isinstance(self.model, AutoModelForSequenceClassification):
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    labels = inputs['labels']
                else:
                    match self.approach:
                        case "tfidf":
                            x, y = batch[0].to(self.device), batch[1].to(self.device)
                            outputs = self.model(x)
                            labels = y
                        case "lstm" | "transformer":
                            inputs = {k: v.to(self.device) for k, v in batch.items()}
                            outputs = self.model(**inputs)
                            labels = inputs["labels"]
                        case _:
                            raise ValueError("Oops! Wrong approach.")

                    outputs = outputs.logits if self.approach == "transformer" else outputs
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

            if self.val_texts:
                val_preds, val_labels = self._predict(val_loader)
                val_acc = accuracy_score(val_labels, val_preds)
                logger.info(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")

        if save_path is not None:
            save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
            save_path.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1, # Save the next epoch number to indicate where to resume
                },
                save_path / "checkpoint.pth"
            )   
        torch.cuda.empty_cache()
        return val_acc or 0.0

    def _predict(self, val_loader: DataLoader):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(self.model, AutoModelForSequenceClassification):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    outputs = self.model(**inputs).logits
                    labels.extend(batch["labels"])
                else:
                    match self.approach:
                        case "tfidf":
                            x, y = batch[0].to(self.device), batch[1].to(self.device)
                            outputs = self.model(x)
                            labels.extend(y)
                        case "lstm" | "transformer":
                            inputs = {k: v.to(self.device) for k, v in batch.items()}
                            outputs = self.model(**inputs)
                            outputs = outputs.logits if self.approach == "transformer" else outputs
                            labels.extend(inputs["labels"])
                        case _:
                            raise ValueError("Oops! Wrong approach.")
                            
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        if isinstance(preds, list):
            preds = [p.item() for p in preds]
            labels = [l.item() for l in labels]
            return np.array(preds), np.array(labels)
        else:
            return preds.cpu().numpy(), labels.cpu().numpy()


    def predict(self, test_data: pd.DataFrame | DataLoader) -> Tuple[np.ndarray, np.ndarray]:

        assert isinstance(test_data, DataLoader) or isinstance(test_data, pd.DataFrame), \
            f"Input data type: {type(test_data)}; Expected: pd.DataFrame | DataLoader"

        if isinstance(test_data, DataLoader):
            return self._predict(test_data)
        
        if self.approach == 'tfidf':
            _X = self.vectorizer.transform(test_data['text'].tolist()).toarray()
            _labels = test_data['label'].tolist()
            _dataset = torch.utils.data.TensorDataset(
                torch.tensor(_X, dtype=torch.float32),
                torch.tensor(_labels)
            )
            _loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=False)
        elif self.approach in ['lstm', 'transformer']:
            _dataset = SimpleTextDataset(
                test_data['text'].tolist(),
                test_data['label'].tolist(),
                self.tokenizer,
                self.token_length
            )
            _loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            raise ValueError(f"Unrecognized approach: {self.approach}")
            # handling any possible tokenization
        
        return self._predict(_loader)


def freeze_layers(model, fraction_layers_to_finetune: float=1.0) -> None:
    total_layers = len(model.distilbert.transformer.layer)
    _num_layers_to_finetune = int(fraction_layers_to_finetune * total_layers)
    layers_to_freeze = total_layers - _num_layers_to_finetune

    for layer in model.distilbert.transformer.layer[:layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False
# end of file