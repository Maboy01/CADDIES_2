#!/usr/bin/env python3
"""
CNN+LSTM video model for GolfDB club classification.

This script reads fixed-length frame sequences from videos_160/<id>.mp4,
encodes each frame with a small CNN, and models the swing over time with an LSTM.
"""

from __future__ import annotations

import argparse
import ast
import copy
import math
from pathlib import Path
import random
import sys
import warnings

import cv2
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as error:
    raise SystemExit(
        "PyTorch is required for train_cnn_lstm.py. Install the compatible build, "
        "for example: pip install torch==2.11.0+cu128 --index-url "
        "https://download.pytorch.org/whl/cu128"
    ) from error


warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "GolfDB.csv"
VIDEOS_DIR = BASE_DIR / "videos_160"
OUTPUT_DIR = BASE_DIR / "model_results" / "cnn_lstm"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
DEFAULT_MAX_VIDEOS = 50
DEFAULT_SEQUENCE_LENGTH = 16
DEFAULT_FRAME_SIZE = 96
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 4
TENSOR_CACHE_VERSION = "event_bbox_rgb_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN+LSTM GolfDB video model.")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=DEFAULT_MAX_VIDEOS,
        help="Maximum videos to load. Use 0 for all videos.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Number of frames sampled per video.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=DEFAULT_FRAME_SIZE,
        help="Square frame size used by the CNN.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=VALIDATION_SIZE,
        help="Fraction of videos reserved for validation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of videos reserved for final test.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience measured in validation epochs.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Use 0 to disable.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="LSTM hidden size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.35,
        help="Dropout used before the classifier.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply lightweight video augmentations to training batches.",
    )
    parser.add_argument(
        "--brightness-jitter",
        type=float,
        default=0.15,
        help="Random brightness jitter used when augmentation is enabled.",
    )
    parser.add_argument(
        "--contrast-jitter",
        type=float,
        default=0.15,
        help="Random contrast jitter used when augmentation is enabled.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Gaussian noise standard deviation used when augmentation is enabled.",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=0.9,
        help="Minimum random crop scale used when augmentation is enabled.",
    )
    parser.add_argument(
        "--use-bbox-crop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Crop frames around the GolfDB bbox before resizing.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.12,
        help="Padding added around the normalized GolfDB bbox crop.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Folder for cached video tensors. Defaults to output-dir/tensor_cache.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild cached video tensors.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Folder where model artifacts and plots are saved.",
    )
    return parser.parse_args()


def set_reproducibility() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(requested_device)


def sample_balanced_rows(dataframe: pd.DataFrame, max_videos: int) -> pd.DataFrame:
    clubs = sorted(dataframe["club"].dropna().unique())
    per_club = max(1, max_videos // len(clubs))
    selected_parts = []
    selected_indexes: set[int] = set()

    for club in clubs:
        group = dataframe[dataframe["club"] == club]
        take = min(len(group), per_club)
        sampled = group.sample(n=take, random_state=RANDOM_STATE)
        selected_parts.append(sampled)
        selected_indexes.update(sampled.index.tolist())

    selected = pd.concat(selected_parts, axis=0)
    remaining = max_videos - len(selected)

    if remaining > 0:
        leftovers = dataframe.drop(index=list(selected_indexes))
        extra = leftovers.sample(
            n=min(remaining, len(leftovers)),
            random_state=RANDOM_STATE,
        )
        selected = pd.concat([selected, extra], axis=0)

    return selected.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def load_metadata(max_videos: int) -> pd.DataFrame:
    print("Loading GolfDB metadata...")
    dataframe = pd.read_csv(CSV_PATH)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.startswith("Unnamed")]
    dataframe["video_path"] = dataframe["id"].apply(
        lambda video_id: VIDEOS_DIR / f"{int(video_id)}.mp4"
    )
    dataframe = dataframe[dataframe["video_path"].apply(lambda path: path.exists())].copy()

    if max_videos > 0 and len(dataframe) > max_videos:
        dataframe = sample_balanced_rows(dataframe, max_videos)

    club_counts = {
        club: int(count)
        for club, count in dataframe["club"].value_counts().sort_index().items()
    }
    print(f"   Videos selected: {len(dataframe)}")
    print(f"   Clubs: {club_counts}")
    return dataframe.reset_index(drop=True)


def parse_events(value: object) -> np.ndarray:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return np.array([], dtype=int)

    if not isinstance(parsed, list):
        return np.array([], dtype=int)

    events = np.array(parsed, dtype=float)
    events = events[np.isfinite(events)]
    return np.rint(events).astype(int)


def parse_bbox(value: object) -> np.ndarray:
    raw = str(value).strip().strip("[]")
    parsed = np.fromstring(raw, sep=" ")
    if len(parsed) != 4 or not np.all(np.isfinite(parsed)):
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    return np.clip(parsed.astype(float), 0.0, 1.0)


def select_evenly(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) == 0:
        return values.astype(int)
    if len(values) <= count:
        return values.astype(int)

    positions = np.linspace(0, len(values) - 1, count).astype(int)
    return values[positions].astype(int)


def build_frame_indexes(
    frame_count: int,
    sequence_length: int,
    events: np.ndarray,
) -> np.ndarray:
    sequence_length = max(2, min(sequence_length, frame_count))
    valid_events = events[(events >= 0) & (events < frame_count)]

    if len(valid_events) >= 2:
        start_frame = int(valid_events[0])
        end_frame = int(valid_events[-1])
    else:
        start_frame = 0
        end_frame = frame_count - 1

    if end_frame <= start_frame:
        start_frame = 0
        end_frame = frame_count - 1

    timeline_indexes = np.linspace(start_frame, end_frame, sequence_length).astype(int)
    anchor_indexes = np.unique(np.concatenate([timeline_indexes, valid_events]))
    selected_indexes = select_evenly(anchor_indexes, sequence_length)

    if len(selected_indexes) < sequence_length:
        pad_count = sequence_length - len(selected_indexes)
        selected_indexes = np.pad(selected_indexes, (0, pad_count), mode="edge")

    return np.clip(selected_indexes, 0, frame_count - 1).astype(int)


def crop_frame_to_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    padding: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        return frame

    box_width = x2 - x1
    box_height = y2 - y1
    x1 = max(0.0, x1 - box_width * padding)
    y1 = max(0.0, y1 - box_height * padding)
    x2 = min(1.0, x2 + box_width * padding)
    y2 = min(1.0, y2 + box_height * padding)

    left = int(round(x1 * width))
    top = int(round(y1 * height))
    right = int(round(x2 * width))
    bottom = int(round(y2 * height))

    if right <= left or bottom <= top:
        return frame

    return frame[top:bottom, left:right]


def load_video_sequence(
    video_path: Path,
    events: np.ndarray,
    bbox: np.ndarray,
    sequence_length: int,
    frame_size: int,
    use_bbox_crop: bool,
    bbox_padding: float,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return np.zeros((sequence_length, 3, frame_size, frame_size), dtype=np.float32)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        return np.zeros((sequence_length, 3, frame_size, frame_size), dtype=np.float32)

    frame_indexes = build_frame_indexes(frame_count, sequence_length, events)
    frames = []
    empty_frame = np.zeros((3, frame_size, frame_size), dtype=np.float32)

    for frame_index in frame_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()

        if not success or frame is None:
            frames.append(empty_frame)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if use_bbox_crop:
            frame = crop_frame_to_bbox(frame, bbox, bbox_padding)
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)

    capture.release()
    return np.stack(frames).astype(np.float32)


def tensor_cache_path(
    cache_dir: Path,
    video_id: int,
    sequence_length: int,
    frame_size: int,
    use_bbox_crop: bool,
    bbox_padding: float,
) -> Path:
    crop_name = "bbox" if use_bbox_crop else "full"
    padding_name = f"pad{int(round(bbox_padding * 100)):03d}"
    filename = (
        f"video_{video_id}_seq{sequence_length}_size{frame_size}_"
        f"{crop_name}_{padding_name}_{TENSOR_CACHE_VERSION}.npy"
    )
    return cache_dir / filename


def build_video_tensors(
    dataframe: pd.DataFrame,
    label_encoder: LabelEncoder,
    sequence_length: int,
    frame_size: int,
    cache_dir: Path,
    refresh_cache: bool,
    use_bbox_crop: bool,
    bbox_padding: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = []
    targets = []
    total_rows = len(dataframe)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for position, row in enumerate(dataframe.itertuples(index=False), start=1):
        cache_path = tensor_cache_path(
            cache_dir,
            int(row.id),
            sequence_length,
            frame_size,
            use_bbox_crop,
            bbox_padding,
        )

        if cache_path.exists() and not refresh_cache:
            print(f"   Loading cached tensor {position}/{total_rows}: {Path(row.video_path).name}")
            sequence = np.load(cache_path).astype(np.float32)
        else:
            print(f"   Reading video {position}/{total_rows}: {Path(row.video_path).name}")
            events = parse_events(row.events)
            bbox = parse_bbox(row.bbox)
            sequence = load_video_sequence(
                Path(row.video_path),
                events,
                bbox,
                sequence_length,
                frame_size,
                use_bbox_crop,
                bbox_padding,
            )
            np.save(cache_path, sequence)

        sequences.append(sequence)
        targets.append(label_encoder.transform([str(row.club)])[0])

    sequence_tensor = torch.from_numpy(np.stack(sequences))
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return sequence_tensor, target_tensor


class CnnLstmClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 128,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = sequences.shape
        frame_batch = sequences.reshape(batch_size * sequence_length, channels, height, width)
        frame_features = self.cnn(frame_batch)
        frame_features = frame_features.flatten(start_dim=1)
        temporal_features = frame_features.reshape(batch_size, sequence_length, -1)
        lstm_output, _hidden_state = self.lstm(temporal_features)
        final_features = lstm_output[:, -1, :]
        return self.classifier(self.dropout(final_features))


def augment_sequence(
    sequence: torch.Tensor,
    brightness_jitter: float,
    contrast_jitter: float,
    noise_std: float,
    crop_scale: float,
) -> torch.Tensor:
    augmented = sequence.clone()

    if random.random() < 0.5:
        augmented = torch.flip(augmented, dims=[3])

    if crop_scale < 1.0 and random.random() < 0.75:
        _time, _channels, height, width = augmented.shape
        scale = random.uniform(crop_scale, 1.0)
        crop_height = max(2, int(height * scale))
        crop_width = max(2, int(width * scale))
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        augmented = augmented[:, :, top : top + crop_height, left : left + crop_width]
        augmented = F.interpolate(
            augmented,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    if brightness_jitter > 0:
        brightness_factor = random.uniform(1 - brightness_jitter, 1 + brightness_jitter)
        augmented = augmented * brightness_factor

    if contrast_jitter > 0:
        contrast_factor = random.uniform(1 - contrast_jitter, 1 + contrast_jitter)
        mean = augmented.mean(dim=(2, 3), keepdim=True)
        augmented = (augmented - mean) * contrast_factor + mean

    if noise_std > 0 and random.random() < 0.5:
        augmented = augmented + torch.randn_like(augmented) * noise_std

    return torch.clamp(augmented, 0.0, 1.0)


class VideoSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        indexes: np.ndarray,
        augment: bool,
        args: argparse.Namespace,
    ) -> None:
        self.sequences = sequences
        self.targets = targets
        self.indexes = indexes.astype(int)
        self.augment = augment
        self.args = args

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_index = int(self.indexes[item_index])
        sequence = self.sequences[source_index]
        target = self.targets[source_index]

        if self.augment:
            sequence = augment_sequence(
                sequence,
                self.args.brightness_jitter,
                self.args.contrast_jitter,
                self.args.noise_std,
                self.args.crop_scale,
            )

        return sequence, target


def split_indices(
    targets: torch.Tensor,
    validation_size: float,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_array = targets.numpy()
    class_counts = pd.Series(target_array).value_counts()
    class_count = len(class_counts)
    can_stratify = class_count > 1 and class_counts.min() >= 3

    if can_stratify:
        test_count = max(math.ceil(len(target_array) * test_size), class_count)
        test_count = min(test_count, len(target_array) - (class_count * 2))
        stratify_targets = target_array
    else:
        test_count = max(1, math.ceil(len(target_array) * test_size))
        stratify_targets = None
        print("   Using non-stratified split because a class has too few videos for 3-way split.")

    all_indexes = np.arange(len(target_array))
    train_validation_indexes, test_indexes = train_test_split(
        all_indexes,
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=stratify_targets,
    )

    train_validation_targets = target_array[train_validation_indexes]
    can_stratify_validation = (
        can_stratify
        and pd.Series(train_validation_targets).value_counts().min() >= 2
    )

    if can_stratify_validation:
        validation_count = max(math.ceil(len(target_array) * validation_size), class_count)
        validation_count = min(validation_count, len(train_validation_indexes) - class_count)
        validation_stratify = train_validation_targets
    else:
        validation_count = max(1, math.ceil(len(target_array) * validation_size))
        validation_stratify = None

    train_indexes, validation_indexes = train_test_split(
        train_validation_indexes,
        test_size=validation_count,
        random_state=RANDOM_STATE,
        stratify=validation_stratify,
    )

    print(
        f"   Split: train={len(train_indexes)}, "
        f"validation={len(validation_indexes)}, test={len(test_indexes)}"
    )
    return train_indexes, validation_indexes, test_indexes


def build_class_weights(targets: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights.to(device)


def make_loader(
    sequences: torch.Tensor,
    targets: torch.Tensor,
    indexes: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    args: argparse.Namespace,
) -> DataLoader:
    dataset = VideoSequenceDataset(sequences, targets, indexes, augment=augment, args=args)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_sequences, batch_targets in loader:
        batch_sequences = batch_sequences.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_sequences)
        loss = criterion(logits, batch_targets)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch_targets.size(0)
        total_loss += float(loss.item()) * batch_size
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += int((predictions == batch_targets).sum().item())
        total_samples += batch_size

    return total_loss / total_samples, correct_predictions / total_samples


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    all_confidences = []

    for batch_sequences, batch_targets in loader:
        batch_sequences = batch_sequences.to(device)
        batch_targets = batch_targets.to(device)
        logits = model(batch_sequences)
        loss = criterion(logits, batch_targets)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)

        batch_size = batch_targets.size(0)
        total_loss += float(loss.item()) * batch_size
        correct_predictions += int((predictions == batch_targets).sum().item())
        total_samples += batch_size
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_targets.extend(batch_targets.cpu().numpy().tolist())
        all_confidences.extend(confidences.cpu().numpy().tolist())

    return (
        total_loss / total_samples,
        correct_predictions / total_samples,
        np.array(all_predictions),
        np.array(all_targets),
        np.array(all_confidences),
    )


def train_cnn_lstm(
    sequences: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]], np.ndarray, np.ndarray, np.ndarray]:
    train_indexes, validation_indexes, test_indexes = split_indices(
        targets,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    train_loader = make_loader(
        sequences,
        targets,
        train_indexes,
        args.batch_size,
        shuffle=True,
        augment=args.augment,
        args=args,
    )
    validation_loader = make_loader(
        sequences,
        targets,
        validation_indexes,
        args.batch_size,
        shuffle=False,
        augment=False,
        args=args,
    )
    test_loader = make_loader(
        sequences,
        targets,
        test_indexes,
        args.batch_size,
        shuffle=False,
        augment=False,
        args=args,
    )

    model = CnnLstmClassifier(
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    class_weights = build_class_weights(targets[train_indexes], num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }
    best_validation_accuracy = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
        )
        (
            validation_loss,
            validation_accuracy,
            _predictions,
            _targets,
            _confidences,
        ) = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
        )
        scheduler.step(validation_accuracy)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best validation accuracy was {best_validation_accuracy:.4f} "
                f"at epoch {best_epoch}."
            )
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = [best_epoch]
    history["best_validation_accuracy"] = [best_validation_accuracy]
    _loss, _accuracy, predictions, test_targets, confidences = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
    )
    return model, history, predictions, test_targets, confidences


def build_metrics(
    test_targets: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    class_names: np.ndarray,
) -> dict[str, object]:
    labels = np.arange(len(class_names))
    accuracy = accuracy_score(test_targets, predictions)
    print(f"\nFinal test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            test_targets,
            predictions,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )

    precision, recall, f1_score, support = precision_recall_fscore_support(
        test_targets,
        predictions,
        labels=labels,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "support": support,
        "test_targets": test_targets,
        "predictions": predictions,
        "confidences": confidences,
        "labels": labels,
        "class_names": class_names,
    }


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(14, 5))

    loss_axis.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#E63946")
    loss_axis.plot(
        epochs,
        history["validation_loss"],
        "s-",
        label="Validation Loss",
        color="#2E86AB",
    )
    loss_axis.set_xlabel("Epoch", fontweight="bold")
    loss_axis.set_ylabel("Loss", fontweight="bold")
    loss_axis.set_title("CNN+LSTM Loss", fontweight="bold")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()

    accuracy_axis.plot(
        epochs,
        history["train_accuracy"],
        "o-",
        label="Train Accuracy",
        color="#06A77D",
    )
    accuracy_axis.plot(
        epochs,
        history["validation_accuracy"],
        "s-",
        label="Validation Accuracy",
        color="#A23B72",
    )
    accuracy_axis.set_xlabel("Epoch", fontweight="bold")
    accuracy_axis.set_ylabel("Accuracy", fontweight="bold")
    accuracy_axis.set_title("CNN+LSTM Accuracy", fontweight="bold")
    accuracy_axis.set_ylim([0, 1.05])
    accuracy_axis.grid(True, alpha=0.3)
    accuracy_axis.legend()

    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(metrics: dict[str, object], output_dir: Path) -> None:
    matrix = confusion_matrix(
        metrics["test_targets"],
        metrics["predictions"],
        labels=metrics["labels"],
    )

    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=metrics["class_names"],
        yticklabels=metrics["class_names"],
        ax=axis,
        cbar_kws={"label": "Count"},
    )
    axis.set_xlabel("Prediction", fontsize=12, fontweight="bold")
    axis.set_ylabel("Actual", fontsize=12, fontweight="bold")
    axis.set_title("CNN+LSTM Confusion Matrix", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_class_metrics(metrics: dict[str, object], output_dir: Path) -> None:
    metric_values = [
        ("Precision", metrics["precision"], "#2E86AB"),
        ("Recall", metrics["recall"], "#A23B72"),
        ("F1-Score", metrics["f1"], "#F18F01"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    for axis, (title, values, color) in zip(axes, metric_values):
        axis.bar(metrics["class_names"], values, color=color, alpha=0.75)
        axis.set_ylabel(title, fontweight="bold")
        axis.set_title(f"{title} by Class", fontweight="bold")
        axis.set_ylim([0, 1])
        axis.tick_params(axis="x", rotation=45)
        axis.grid(True, alpha=0.3, axis="y")

    figure.tight_layout()
    figure.savefig(output_dir / "metrics_by_class.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_prediction_distribution(metrics: dict[str, object], output_dir: Path) -> None:
    labels = np.asarray(metrics["labels"])
    class_names = np.asarray(metrics["class_names"])
    test_targets = np.asarray(metrics["test_targets"])
    predictions = np.asarray(metrics["predictions"])

    actual_counts = np.array([(test_targets == label).sum() for label in labels])
    predicted_counts = np.array([(predictions == label).sum() for label in labels])

    x_positions = np.arange(len(class_names))
    bar_width = 0.38
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(
        x_positions - bar_width / 2,
        actual_counts,
        width=bar_width,
        label="Actual",
        color="#2E86AB",
        alpha=0.8,
    )
    axis.bar(
        x_positions + bar_width / 2,
        predicted_counts,
        width=bar_width,
        label="Predicted",
        color="#F18F01",
        alpha=0.8,
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(class_names, rotation=35, ha="right")
    axis.set_ylabel("Samples", fontweight="bold")
    axis.set_title("Actual vs Predicted Distribution", fontweight="bold")
    axis.grid(True, alpha=0.3, axis="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confidence_distribution(metrics: dict[str, object], output_dir: Path) -> None:
    confidences = np.asarray(metrics["confidences"], dtype=float)
    test_targets = np.asarray(metrics["test_targets"])
    predictions = np.asarray(metrics["predictions"])
    correct_mask = test_targets == predictions
    bins = np.linspace(0, 1, 11)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(
        confidences[correct_mask],
        bins=bins,
        alpha=0.75,
        label="Correct",
        color="#06A77D",
        edgecolor="white",
    )
    axis.hist(
        confidences[~correct_mask],
        bins=bins,
        alpha=0.75,
        label="Incorrect",
        color="#E63946",
        edgecolor="white",
    )
    axis.set_xlabel("Prediction Confidence", fontweight="bold")
    axis.set_ylabel("Samples", fontweight="bold")
    axis.set_title("Confidence Distribution", fontweight="bold")
    axis.set_xlim([0, 1])
    axis.grid(True, alpha=0.3, axis="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "confidence_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_training_tables(
    history: dict[str, list[float]],
    metrics: dict[str, object],
    output_dir: Path,
) -> None:
    history_frame = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "train_accuracy": history["train_accuracy"],
            "validation_loss": history["validation_loss"],
            "validation_accuracy": history["validation_accuracy"],
            "learning_rate": history["learning_rate"],
        }
    )
    history_frame.to_csv(output_dir / "training_history.csv", index=False)

    class_names = np.asarray(metrics["class_names"])
    class_metrics_frame = pd.DataFrame(
        {
            "class": class_names,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "support": metrics["support"],
        }
    )
    class_metrics_frame.to_csv(output_dir / "class_metrics.csv", index=False)

    test_targets = np.asarray(metrics["test_targets"], dtype=int)
    predictions = np.asarray(metrics["predictions"], dtype=int)
    confidences = np.asarray(metrics["confidences"], dtype=float)
    predictions_frame = pd.DataFrame(
        {
            "actual_index": test_targets,
            "predicted_index": predictions,
            "actual_class": class_names[test_targets],
            "predicted_class": class_names[predictions],
            "confidence": confidences,
            "correct": test_targets == predictions,
        }
    )
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)

    summary_frame = pd.DataFrame(
        [
            {
                "test_accuracy": metrics["accuracy"],
                "best_epoch": history["best_epoch"][0],
                "best_validation_accuracy": history["best_validation_accuracy"][0],
                "epochs_ran": len(history["train_loss"]),
            }
        ]
    )
    summary_frame.to_csv(output_dir / "run_summary.csv", index=False)


def save_model(
    model: nn.Module,
    label_encoder: LabelEncoder,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": label_encoder.classes_.tolist(),
        "sequence_length": args.sequence_length,
        "frame_size": args.frame_size,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "model": "cnn_lstm",
    }
    torch.save(checkpoint, output_dir / "cnn_lstm_model.pt")


def main() -> None:
    args = parse_args()
    if args.max_videos < 0:
        raise ValueError("--max-videos must be 0 or greater.")
    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be at least 2.")
    if args.frame_size < 32:
        raise ValueError("--frame-size must be at least 32.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if not 0 < args.validation_size < 0.5:
        raise ValueError("--validation-size must be between 0 and 0.5.")
    if not 0 < args.test_size < 0.5:
        raise ValueError("--test-size must be between 0 and 0.5.")
    if args.validation_size + args.test_size >= 0.8:
        raise ValueError("--validation-size plus --test-size must leave enough training data.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be between 0 and 1.")
    if not 0 < args.crop_scale <= 1:
        raise ValueError("--crop-scale must be greater than 0 and at most 1.")
    if args.bbox_padding < 0:
        raise ValueError("--bbox-padding must be 0 or greater.")
    if args.cache_dir is None:
        args.cache_dir = args.output_dir / "tensor_cache"

    set_reproducibility()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print("\nCNN+LSTM Golf Swing Training")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"Max videos: {args.max_videos if args.max_videos else 'all'}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Frame size: {args.frame_size}")
    print(f"Augmentation: {args.augment}")
    print(f"BBox crop: {args.use_bbox_crop}")
    print(f"Tensor cache: {args.cache_dir}\n")

    dataframe = load_metadata(args.max_videos)
    label_encoder = LabelEncoder()
    label_encoder.fit(dataframe["club"].astype(str))
    sequences, targets = build_video_tensors(
        dataframe,
        label_encoder,
        args.sequence_length,
        args.frame_size,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        use_bbox_crop=args.use_bbox_crop,
        bbox_padding=args.bbox_padding,
    )

    model, history, predictions, test_targets, confidences = train_cnn_lstm(
        sequences,
        targets,
        num_classes=len(label_encoder.classes_),
        args=args,
        device=device,
    )
    metrics = build_metrics(test_targets, predictions, confidences, label_encoder.classes_)

    print("Saving CNN+LSTM artifacts...")
    plot_history(history, args.output_dir)
    plot_confusion_matrix(metrics, args.output_dir)
    plot_class_metrics(metrics, args.output_dir)
    plot_prediction_distribution(metrics, args.output_dir)
    plot_confidence_distribution(metrics, args.output_dir)
    save_training_tables(history, metrics, args.output_dir)
    save_model(model, label_encoder, args, args.output_dir)

    print("\nCNN+LSTM training completed.")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
