from __future__ import annotations

import logging
import os
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from wisent_guard.core.utils.device import resolve_default_device

__all__ = [
    "ClassifierError",
    "ClassifierTrainConfig",
    "ClassifierMetrics",
    "ClassifierTrainReport",
    "BaseClassifier",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ClassifierTrainConfig:
    """Training hyperparameters."""
    test_size: float = 0.2
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    random_state: int = 42


@dataclass(slots=True, frozen=True)
class ClassifierMetrics:
    """Common binary classification ClassifierMetrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


@dataclass(slots=True, frozen=True)
class ClassifierTrainReport:
    classifier_name: str
    input_dim: int
    best_epoch: int
    epochs_ran: int
    final: ClassifierMetrics
    history: dict[str, list[float]]

    def asdict(self) -> dict[str, Any]:
        d = asdict(self)
        d["final"] = asdict(self.final)
        return d

class ClassifierError(RuntimeError):
    """Raised when a classifier cannot complete its task."""


class BaseClassifier(ABC):
    """
    Abstract base class for binary classifiers on dense vectors.
    Subclasses must implement build_model() to return a torch.nn.Module
    that outputs probabilities (0.0 to 1.0) for the positive class.

    attributes:
        name: str - unique name of the classifier type (e.g. "logistic_regression")
        description: str - human-readable description
        model: nn.Module | None - the PyTorch model (None if not yet built)
        device: str - device for model and tensors ("cpu", "cuda", "mps", etc)
        dtype: torch.dtype - floating point type for tensors (torch.float32, torch.float64, etc)
        threshold: float - decision threshold for positive class (default 0.5)
    """
    name: str = "base"
    description: str = "Abstract classifier"

    _REGISTRY: dict[str, type["BaseClassifier"]] = {}

    model: nn.Module | None
    device: str
    dtype: torch.dtype
    threshold: float

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls is BaseClassifier:
            return
        if not getattr(cls, "name", None):
            raise TypeError("Classifier subclasses must define a class attribute `name`.")
        if cls.name in BaseClassifier._REGISTRY:
            raise ValueError(f"Duplicate classifier name: {cls.name!r}")
        BaseClassifier._REGISTRY[cls.name] = cls

    def __init__(
        self,
        threshold: float = 0.5,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        self.threshold = float(threshold)
        self.device = self.select_device(device)
        self.dtype = torch.float32 if self.device == "mps" else dtype
        self.model = None

    @abstractmethod
    def build_model(self, input_dim: int, **model_params: Any) -> nn.Module:
        """Return a torch.nn.Module implementing binary classification."""
        raise NotImplementedError

    def model_hyperparams(self) -> dict[str, Any]:
        return {}

    def fit(self, X, y, /, *, config: ClassifierTrainConfig | None = None, **model_params: Any) -> ClassifierTrainReport:
        """
        Train the classifier on dense vectors (list/ndarray/tensor of shape [N, D]).
        """
        cfg = config or ClassifierTrainConfig()
        torch.manual_seed(cfg.random_state)

        X_tensor = self.to_2d_tensor(X, device=self.device, dtype=self.dtype)
        y_tensor = self.to_1d_tensor(y, device=self.device, dtype=self.dtype)
        if X_tensor.shape[0] != y_tensor.shape[0]:
            raise ClassifierError(f"X and y length mismatch: {X_tensor.shape[0]} vs {y_tensor.shape[0]}")

        input_dim = int(X_tensor.shape[1])
        if self.model is None:
            self.model = self.build_model(input_dim, **model_params).to(self.device)

        train_loader, test_loader = self.make_dataloaders(X_tensor, y_tensor, cfg)

        criterion = self.configure_criterion()
        optimizer = self.configure_optimizer(self.model, cfg.learning_rate)

        best_acc = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0

        history: dict[str, list[float]] = {
            "train_loss": [], "test_loss": [],
            "accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [],
        }

        for epoch in range(cfg.num_epochs):
            train_loss = self.train_one_epoch(self.model, train_loader, optimizer, criterion)
            test_loss, probs, labels = self.eval_one_epoch(self.model, test_loader, criterion)

            preds = [1.0 if p >= self.threshold else 0.0 for p in probs]
            acc, prec, rec, f1 = self.basic_prf(preds, labels)
            auc = self.roc_auc(labels, probs)

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["accuracy"].append(acc)
            history["precision"].append(prec)
            history["recall"].append(rec)
            history["f1"].append(f1)
            history["auc"].append(auc)

            if (epoch == 0) or ((epoch + 1) % 10 == 0) or (epoch == cfg.num_epochs - 1):
                logger.info(
                    "[%s] epoch %d/%d  train=%.4f  test=%.4f  acc=%.4f  f1=%.4f",
                    self.name, epoch + 1, cfg.num_epochs, train_loss, test_loss, acc, f1
                )

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    logger.info("[%s] early stopping at epoch %d", self.name, epoch + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        test_loss, probs, labels = self.eval_one_epoch(self.model, test_loader, criterion)
        preds = [1.0 if p >= self.threshold else 0.0 for p in probs]
        acc, prec, rec, f1 = self.basic_prf(preds, labels)
        auc = self.roc_auc(labels, probs)
        final = ClassifierMetrics(acc, prec, rec, f1, auc)

        best_epoch = int(max(range(len(history["accuracy"])), key=history["accuracy"].__getitem__) + 1)
        report = ClassifierTrainReport(
            classifier_name=self.name,
            input_dim=input_dim,
            best_epoch=best_epoch,
            epochs_ran=len(history["accuracy"]),
            final=final,
            history={k: [float(v) for v in vs] for k, vs in history.items()},
        )
        return report

    def predict(self, X) -> int | list[int]:
        """Predict {0,1} labels; returns int for a single sample, list for batches."""
        self.require_model()
        X_tensor = self.to_2d_tensor(X, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            probs = self.forward_probs(self.model, X_tensor).view(-1).cpu().tolist()
        preds = [1 if p >= self.threshold else 0 for p in probs]
        return preds[0] if len(preds) == 1 else preds

    def predict_proba(self, X) -> float | list[float]:
        """Return P(y=1) as float or list[float]."""
        self.require_model()
        X_tensor = self.to_2d_tensor(X, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            probs = self.forward_probs(self.model, X_tensor).view(-1).cpu().tolist()
        return probs[0] if len(probs) == 1 else probs

    def evaluate(self, X, y) -> dict[str, float]:
        """Compute accuracy/precision/recall/F1/AUC on provided data."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        if isinstance(y_pred, int):
            preds = [float(y_pred)]
            probs = [float(y_prob)]  
        else:
            preds = [float(v) for v in y_pred]
            probs = [float(v) for v in y_prob]  

        if isinstance(y, torch.Tensor):
            labels = y.detach().cpu().view(-1).tolist()
        else:
            labels = list(y)

        acc, prec, rec, f1 = self.basic_prf(preds, labels)
        auc = self.roc_auc(labels, probs)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

    def make_dataloaders(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        cfg: ClassifierTrainConfig,
    ) -> tuple[DataLoader, DataLoader]:
        """Split into train/test and wrap in DataLoaders."""
        ds = TensorDataset(X, y)
        if len(ds) < 2:
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True), DataLoader(ds, batch_size=cfg.batch_size)

        test_count = max(1, int(round(cfg.test_size * len(ds))))
        test_count = min(test_count, len(ds) - 1)
        train_count = len(ds) - test_count
        train_ds, test_ds = random_split(ds, [train_count, test_count], generator=torch.Generator().manual_seed(cfg.random_state))
        return (
            DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False),
        )

    def configure_criterion(self) -> nn.Module:
        return nn.BCELoss()

    def configure_optimizer(self, model: nn.Module, lr: float) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=lr)

    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        model.train()
        total = 0.0
        steps = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            probs = self.forward_probs(model, xb)
            loss = criterion(probs.view(-1), yb.view(-1))
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            steps += 1
        return total / max(steps, 1)

    def eval_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, list[float], list[float]]:
        model.eval()
        total = 0.0
        steps = 0
        all_probs: list[float] = []
        all_labels: list[float] = []
        with torch.no_grad():
            for xb, yb in loader:
                probs = self.forward_probs(model, xb)
                loss = criterion(probs.view(-1), yb.view(-1))
                total += float(loss.item())
                steps += 1
                all_probs.extend(probs.detach().cpu().view(-1).tolist())
                all_labels.extend(yb.detach().cpu().view(-1).tolist())
        return (total / max(steps, 1), all_probs, all_labels)

    def forward_probs(self, model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probabilities shaped (N, 1)."""
        if xb.device.type != self.device:
            xb = xb.to(self.device)
        if xb.dtype != self.dtype:
            xb = xb.to(self.dtype)
        out = model(xb)
        return out.view(-1, 1) if out.ndim == 1 else out

    def save_model(self, path: str) -> None:
        self.require_model()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        input_dim = int(next(self.model.parameters()).shape[1])
        payload = {
            "classifier_name": self.name,
            "state_dict": self.model.state_dict(),
            "input_dim": input_dim,
            "threshold": self.threshold,
            "model_hyperparams": self.model_hyperparams(),
        }
        torch.save(payload, path)
        logger.info("Saved %s to %s", self.name, path)

    def load_model(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(data, dict) or "state_dict" not in data or "input_dim" not in data:
            raise ClassifierError("Unsupported checkpoint format.")
        self.threshold = float(data.get("threshold", self.threshold))
        input_dim = int(data["input_dim"])
        hyper = dict(data.get("model_hyperparams", {}))
        self.model = self.build_model(input_dim, **hyper).to(self.device)
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()

    def set_threshold(self, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        self.threshold = float(threshold)

    def require_model(self) -> None:
        if self.model is None:
            raise ClassifierError("Model not initialized. Call fit() or load_model() first.")

    @staticmethod
    def select_device(device: str | None) -> str:
        if device:
            return device
        return resolve_default_device()

    @classmethod
    def to_2d_tensor(cls, X, *, device: str, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            t = X.to(device=device, dtype=dtype)
            if t.ndim == 1:
                t = t.view(1, -1)
            if t.ndim != 2:
                raise ClassifierError(f"Expected 2D features, got shape {tuple(t.shape)}")
            return t
        # list/iterable-of-vectors
        t = torch.tensor(X, device=device, dtype=dtype)
        if t.ndim == 1:
            t = t.view(1, -1)
        if t.ndim != 2:
            raise ClassifierError(f"Expected 2D features, got shape {tuple(t.shape)}")
        return t

    @staticmethod
    def to_1d_tensor(y, *, device: str, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(y, torch.Tensor):
            return y.to(device=device, dtype=dtype).view(-1)
        return torch.tensor(list(y), device=device, dtype=dtype).view(-1)

    @staticmethod
    def normalize_text(s: str) -> str:
        s2 = unicodedata.normalize("NFKD", s)
        s2 = "".join(ch for ch in s2 if not unicodedata.combining(ch))
        s2 = s2.lower()
        s2 = re.sub(r"[^\w\s]", " ", s2)
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2

    @staticmethod
    def basic_prf(preds: list[float], labels: list[float]) -> tuple[float, float, float, float]:
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        total = max(len(labels), 1)
        acc = sum(1 for p, l in zip(preds, labels) if p == l) / total
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return float(acc), float(prec), float(rec), float(f1)

    @staticmethod
    def roc_auc(labels: list[float], scores: list[float]) -> float:
        """Dependency-free ROC AUC with tie handling."""
        if len(scores) < 2 or len(set(labels)) < 2:
            return 0.0
        pairs = sorted(zip(scores, labels), key=lambda x: x[0])
        pos = sum(1 for _, y in pairs if y == 1)
        neg = sum(1 for _, y in pairs if y == 0)
        if pos == 0 or neg == 0:
            return 0.0
        rank_sum = 0.0
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
                j += 1
            avg_rank = (i + j + 2) / 2.0  # 1-based
            rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1] == 1)
            i = j + 1
        U = rank_sum - pos * (pos + 1) / 2.0
        return float(U / (pos * neg))

    @classmethod
    def list_registered(cls) -> dict[str, type["BaseClassifier"]]:
        return dict(cls._REGISTRY)

    @classmethod
    def get(cls, name: str) -> type["BaseClassifier"]:
        try:
            return cls._REGISTRY[name]
        except KeyError as exc:
            raise ClassifierError(f"Unknown classifier: {name!r}") from exc