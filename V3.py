#!/usr/bin/env python3
"""AICUP table-tennis sequence training and submission pipeline.

The test file contains the first n-1 strokes of each rally.  This script turns
complete/labelled rallies into prefix examples:

    strokes[0:t] -> strokes[t].actionId, strokes[t].pointId, serverGetPoint

It then trains a GRU model and blends it with frequency backoff models.  The
default neural version uses separate single-task models for action, point, and
server so the two Macro-F1 tasks can tune independently.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


ACTION_CLASSES = 15  # next stroke is after an observed serve, so serve actions 15-18 are not normal targets
POINT_CLASSES = 10

ROW_CAT_COLS = [
    "sex",
    "match",
    "numberGame",
    "rally_id",
    "scoreSelf",
    "scoreOther",
    "scoreDiffBucket",
    "scoreTotalBucket",
    "gamePlayerId",
    "gamePlayerOtherId",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
    "serverPlayerId",
    "receiverPlayerId",
    "isServerHitter",
]

ROW_NUM_COLS = [
    "strikeNumberNorm",
    "scoreSelfNorm",
    "scoreOtherNorm",
    "scoreDiffNorm",
    "scoreTotalNorm",
]

QUERY_CAT_COLS = [
    "sex",
    "match",
    "numberGame",
    "rally_id",
    "scoreSelf",
    "scoreOther",
    "scoreDiffBucket",
    "scoreTotalBucket",
    "serverPlayerId",
    "receiverPlayerId",
    "prefixLenBucket",
    "nextStrikeId",
    "nextStrikeNumberBucket",
    "nextHitterId",
    "nextOpponentId",
    "nextHitterIsServer",
    "lastStrikeId",
    "lastHandId",
    "lastStrengthId",
    "lastSpinId",
    "lastPointId",
    "lastActionId",
    "lastPositionId",
    "prevStrikeId",
    "prevHandId",
    "prevStrengthId",
    "prevSpinId",
    "prevPointId",
    "prevActionId",
    "prevPositionId",
    "prev2PointId",
    "prev2ActionId",
    "firstActionId",
    "firstPointId",
    "firstSpinId",
]

QUERY_NUM_COLS = [
    "prefixLenNorm",
    "nextStrikeNumberNorm",
    "scoreSelfNorm",
    "scoreOtherNorm",
    "scoreDiffNorm",
    "scoreTotalNorm",
]

ACTION_KEY_SPECS = [
    (("prefixLenBucket", "nextStrikeId", "lastActionId", "lastPointId", "prevActionId", "prevPointId", "sex"), 4.0, 16.0),
    (("nextStrikeId", "lastActionId", "prevActionId", "lastPointId", "prevPointId"), 3.0, 14.0),
    (("prefixLenBucket", "nextStrikeId", "lastActionId", "lastPointId", "lastSpinId", "lastHandId", "sex"), 3.0, 12.0),
    (("prefixLenBucket", "nextStrikeId", "lastActionId", "lastPointId"), 2.5, 10.0),
    (("nextStrikeId", "lastActionId", "lastPointId", "lastSpinId"), 2.0, 10.0),
    (("lastActionId", "prevActionId", "prev2ActionId"), 1.8, 12.0),
    (("prefixLenBucket", "nextStrikeId", "lastActionId", "lastHandId"), 1.6, 8.0),
    (("nextStrikeId", "lastActionId", "lastPointId"), 1.6, 8.0),
    (("nextStrikeId", "lastActionId"), 1.2, 6.0),
    (("prefixLenBucket", "nextStrikeId", "sex"), 0.9, 5.0),
    (("nextStrikeId", "sex"), 0.7, 5.0),
    (("nextStrikeId",), 0.6, 4.0),
]

POINT_KEY_SPECS = [
    (("prefixLenBucket", "nextStrikeId", "lastPointId", "prevPointId", "lastActionId", "prevActionId", "sex"), 4.0, 16.0),
    (("nextStrikeId", "lastPointId", "prevPointId", "lastActionId", "prevActionId"), 3.0, 14.0),
    (("prefixLenBucket", "nextStrikeId", "lastActionId", "lastPointId", "lastSpinId", "sex"), 3.0, 12.0),
    (("prefixLenBucket", "nextStrikeId", "lastPointId", "lastActionId"), 2.4, 10.0),
    (("nextStrikeId", "lastPointId", "lastActionId", "lastPositionId"), 1.8, 8.0),
    (("lastPointId", "prevPointId", "prev2PointId"), 1.8, 12.0),
    (("nextStrikeId", "lastPointId", "lastSpinId"), 1.6, 8.0),
    (("nextStrikeId", "lastPointId"), 1.3, 6.0),
    (("prefixLenBucket", "nextStrikeId", "sex"), 0.9, 5.0),
    (("nextStrikeId", "sex"), 0.7, 5.0),
    (("nextStrikeId",), 0.6, 4.0),
]

SERVER_KEY_SPECS = [
    (("sex", "scoreSelf", "scoreOther", "prefixLenBucket", "serverPlayerId", "receiverPlayerId"), 3.0, 12.0),
    (("serverPlayerId", "receiverPlayerId", "prefixLenBucket"), 2.3, 10.0),
    (("sex", "scoreSelf", "scoreOther", "prefixLenBucket"), 1.8, 8.0),
    (("sex", "lastActionId", "lastPointId", "prefixLenBucket"), 1.5, 8.0),
    (("nextHitterIsServer", "lastActionId", "lastPointId"), 1.2, 6.0),
    (("scoreDiffBucket", "prefixLenBucket", "sex"), 1.0, 6.0),
    (("serverPlayerId", "receiverPlayerId"), 0.9, 8.0),
    (("sex",), 0.5, 4.0),
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def add_derived(df: pd.DataFrame, has_server: bool) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    first_player = df.groupby("rally_uid")["gamePlayerId"].transform("first")
    first_other = df.groupby("rally_uid")["gamePlayerOtherId"].transform("first")
    df["serverPlayerId"] = first_player.astype(int)
    df["receiverPlayerId"] = first_other.astype(int)
    df["isServerHitter"] = (df["gamePlayerId"] == df["serverPlayerId"]).astype(int)
    df["scoreDiff"] = (df["scoreSelf"] - df["scoreOther"]).astype(int)
    df["scoreTotal"] = (df["scoreSelf"] + df["scoreOther"]).astype(int)
    df["scoreDiffBucket"] = df["scoreDiff"].clip(-15, 15).astype(int)
    df["scoreTotalBucket"] = df["scoreTotal"].clip(0, 40).astype(int)
    df["strikeNumberNorm"] = df["strikeNumber"].astype(float) / 60.0
    df["scoreSelfNorm"] = df["scoreSelf"].astype(float) / 25.0
    df["scoreOtherNorm"] = df["scoreOther"].astype(float) / 25.0
    df["scoreDiffNorm"] = df["scoreDiff"].astype(float) / 25.0
    df["scoreTotalNorm"] = df["scoreTotal"].astype(float) / 50.0
    if has_server:
        df["serverGetPoint"] = df["serverGetPoint"].astype(int)
    return df


def next_strike_id(prefix_len: int) -> int:
    next_no = prefix_len + 1
    if next_no == 1:
        return 1
    if next_no == 2:
        return 2
    return 4


def prefix_len_bucket(prefix_len: int) -> int:
    return int(min(prefix_len, 16))


def make_query_record(group: pd.DataFrame, prefix_len: int) -> Dict[str, float]:
    first = group.iloc[0]
    last = group.iloc[prefix_len - 1]
    prev = group.iloc[prefix_len - 2] if prefix_len >= 2 else None
    prev2 = group.iloc[prefix_len - 3] if prefix_len >= 3 else None
    score_diff = int(last["scoreSelf"] - last["scoreOther"])
    score_total = int(last["scoreSelf"] + last["scoreOther"])
    next_no = prefix_len + 1
    next_hitter = int(last["gamePlayerOtherId"])
    next_opponent = int(last["gamePlayerId"])
    rec = {
        "rally_uid": int(last["rally_uid"]),
        "sex": int(last["sex"]),
        "match": int(last["match"]),
        "numberGame": int(last["numberGame"]),
        "rally_id": int(last["rally_id"]),
        "scoreSelf": int(last["scoreSelf"]),
        "scoreOther": int(last["scoreOther"]),
        "scoreDiffBucket": int(np.clip(score_diff, -15, 15)),
        "scoreTotalBucket": int(np.clip(score_total, 0, 40)),
        "serverPlayerId": int(first["gamePlayerId"]),
        "receiverPlayerId": int(first["gamePlayerOtherId"]),
        "prefix_len": int(prefix_len),
        "prefixLenBucket": prefix_len_bucket(prefix_len),
        "nextStrikeId": next_strike_id(prefix_len),
        "nextStrikeNumberBucket": int(min(next_no, 17)),
        "nextHitterId": next_hitter,
        "nextOpponentId": next_opponent,
        "nextHitterIsServer": int(next_hitter == int(first["gamePlayerId"])),
        "lastStrikeId": int(last["strikeId"]),
        "lastHandId": int(last["handId"]),
        "lastStrengthId": int(last["strengthId"]),
        "lastSpinId": int(last["spinId"]),
        "lastPointId": int(last["pointId"]),
        "lastActionId": int(last["actionId"]),
        "lastPositionId": int(last["positionId"]),
        "prevStrikeId": int(prev["strikeId"]) if prev is not None else -1,
        "prevHandId": int(prev["handId"]) if prev is not None else -1,
        "prevStrengthId": int(prev["strengthId"]) if prev is not None else -1,
        "prevSpinId": int(prev["spinId"]) if prev is not None else -1,
        "prevPointId": int(prev["pointId"]) if prev is not None else -1,
        "prevActionId": int(prev["actionId"]) if prev is not None else -1,
        "prevPositionId": int(prev["positionId"]) if prev is not None else -1,
        "prev2PointId": int(prev2["pointId"]) if prev2 is not None else -1,
        "prev2ActionId": int(prev2["actionId"]) if prev2 is not None else -1,
        "firstActionId": int(first["actionId"]),
        "firstPointId": int(first["pointId"]),
        "firstSpinId": int(first["spinId"]),
        "prefixLenNorm": float(prefix_len) / 60.0,
        "nextStrikeNumberNorm": float(next_no) / 60.0,
        "scoreSelfNorm": float(last["scoreSelf"]) / 25.0,
        "scoreOtherNorm": float(last["scoreOther"]) / 25.0,
        "scoreDiffNorm": float(score_diff) / 25.0,
        "scoreTotalNorm": float(score_total) / 50.0,
    }
    return rec


def build_transition_meta(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for _, group in df.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber")
        if len(group) < 2:
            continue
        server_label = int(group["serverGetPoint"].iloc[0])
        for prefix_len in range(1, len(group)):
            target = group.iloc[prefix_len]
            target_action = int(target["actionId"])
            if target_action >= ACTION_CLASSES:
                continue
            rec = make_query_record(group, prefix_len)
            rec["target_action"] = target_action
            rec["target_point"] = int(target["pointId"])
            rec["target_server"] = server_label
            rec["source"] = source
            rows.append(rec)
    return pd.DataFrame(rows)


def build_observed_transition_meta(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Build action/point labels from observed strokes in an unlabelled test prefix."""
    rows: List[Dict[str, float]] = []
    for _, group in df.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber")
        if len(group) < 2:
            continue
        for prefix_len in range(1, len(group)):
            target = group.iloc[prefix_len]
            target_action = int(target["actionId"])
            if target_action >= ACTION_CLASSES:
                continue
            rec = make_query_record(group, prefix_len)
            rec["target_action"] = target_action
            rec["target_point"] = int(target["pointId"])
            rec["target_server"] = 0
            rec["source"] = source
            rows.append(rec)
    return pd.DataFrame(rows)


def build_test_meta(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for _, group in df.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber")
        rows.append(make_query_record(group, len(group)))
    return pd.DataFrame(rows)


class CategoryEncoder:
    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)
        self.maps: Dict[str, Dict[int, int]] = {}
        self.sizes: Dict[str, int] = {}

    def fit(self, frames: Sequence[pd.DataFrame]) -> "CategoryEncoder":
        for col in self.columns:
            values: set[int] = set()
            for frame in frames:
                if col in frame.columns:
                    values.update(int(v) for v in frame[col].dropna().unique())
            self.maps[col] = {v: i + 1 for i, v in enumerate(sorted(values))}
            self.sizes[col] = len(self.maps[col]) + 1
        return self

    def transform_frame(self, frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        arrays = []
        for col in columns:
            mapping = self.maps[col]
            arrays.append(frame[col].map(lambda x: mapping.get(int(x), 0)).to_numpy(np.int64))
        return np.stack(arrays, axis=1) if arrays else np.zeros((len(frame), 0), dtype=np.int64)

    def cardinalities(self, columns: Sequence[str]) -> List[int]:
        return [self.sizes[col] for col in columns]


class SequenceDataset:
    def __init__(self, stroke_df: pd.DataFrame, meta: pd.DataFrame, encoder: CategoryEncoder):
        self.meta = meta.reset_index(drop=True)
        self.encoder = encoder
        self.row_cat: Dict[int, np.ndarray] = {}
        self.row_num: Dict[int, np.ndarray] = {}
        for uid, group in stroke_df.groupby("rally_uid", sort=False):
            group = group.sort_values("strikeNumber")
            self.row_cat[int(uid)] = encoder.transform_frame(group, ROW_CAT_COLS)
            self.row_num[int(uid)] = group[ROW_NUM_COLS].to_numpy(np.float32)
        self.query_cat = encoder.transform_frame(self.meta, QUERY_CAT_COLS)
        self.query_num = self.meta[QUERY_NUM_COLS].to_numpy(np.float32)
        self.y_action = self.meta["target_action"].to_numpy(np.int64) if "target_action" in meta else None
        self.y_point = self.meta["target_point"].to_numpy(np.int64) if "target_point" in meta else None
        self.y_server = self.meta["target_server"].to_numpy(np.float32) if "target_server" in meta else None
        self.uids = self.meta["rally_uid"].to_numpy(np.int64)
        self.prefix_lens = self.meta["prefix_len"].to_numpy(np.int64)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        uid = int(self.uids[idx])
        plen = int(self.prefix_lens[idx])
        item = {
            "seq_cat": self.row_cat[uid][:plen],
            "seq_num": self.row_num[uid][:plen],
            "query_cat": self.query_cat[idx],
            "query_num": self.query_num[idx],
            "uid": uid,
        }
        if self.y_action is not None:
            item["y_action"] = self.y_action[idx]
            item["y_point"] = self.y_point[idx]
            item["y_server"] = self.y_server[idx]
        return item


def make_collate_fn():
    import torch

    def collate(batch):
        batch_size = len(batch)
        max_len = max(x["seq_cat"].shape[0] for x in batch)
        n_row_cat = batch[0]["seq_cat"].shape[1]
        n_row_num = batch[0]["seq_num"].shape[1]
        seq_cat = torch.zeros(batch_size, max_len, n_row_cat, dtype=torch.long)
        seq_num = torch.zeros(batch_size, max_len, n_row_num, dtype=torch.float32)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        for i, item in enumerate(batch):
            length = item["seq_cat"].shape[0]
            lengths[i] = length
            seq_cat[i, :length] = torch.as_tensor(item["seq_cat"], dtype=torch.long)
            seq_num[i, :length] = torch.as_tensor(item["seq_num"], dtype=torch.float32)
        out = {
            "seq_cat": seq_cat,
            "seq_num": seq_num,
            "lengths": lengths,
            "query_cat": torch.as_tensor(np.stack([x["query_cat"] for x in batch]), dtype=torch.long),
            "query_num": torch.as_tensor(np.stack([x["query_num"] for x in batch]), dtype=torch.float32),
            "uid": np.array([x["uid"] for x in batch], dtype=np.int64),
        }
        if "y_action" in batch[0]:
            out["y_action"] = torch.as_tensor([x["y_action"] for x in batch], dtype=torch.long)
            out["y_point"] = torch.as_tensor([x["y_point"] for x in batch], dtype=torch.long)
            out["y_server"] = torch.as_tensor([x["y_server"] for x in batch], dtype=torch.float32)
        return out

    return collate


def embedding_dim(cardinality: int) -> int:
    if cardinality <= 2:
        return 2
    return int(min(32, max(3, round(1.6 * math.sqrt(cardinality)))))


def class_weights(y: np.ndarray, n_classes: int, power: float = 0.45) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64) + 1.0
    weights = (counts.mean() / counts) ** power
    weights = weights / weights.mean()
    return np.clip(weights, 0.35, 4.0).astype(np.float32)


def build_sequence_backbone(
    row_cards: List[int],
    query_cards: List[int],
    row_num_dim: int,
    query_num_dim: int,
    hidden: int,
    dropout: float,
):
    import torch
    import torch.nn as nn

    class CatBlock(nn.Module):
        def __init__(self, cards: List[int]):
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(card, embedding_dim(card), padding_idx=0) for card in cards]
            )
            self.out_dim = sum(embedding_dim(card) for card in cards)

        def forward(self, x):
            return torch.cat([emb(x[..., i]) for i, emb in enumerate(self.embeddings)], dim=-1)

    class SequenceBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.row_cat = CatBlock(row_cards)
            self.query_cat = CatBlock(query_cards)
            row_in = self.row_cat.out_dim + row_num_dim
            query_in = self.query_cat.out_dim + query_num_dim
            self.row_proj = nn.Sequential(
                nn.Linear(row_in, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.gru = nn.GRU(
                input_size=hidden,
                hidden_size=hidden,
                num_layers=2,
                dropout=dropout,
                batch_first=True,
                bidirectional=True,
            )
            self.query_proj = nn.Sequential(
                nn.Linear(query_in, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            shared_in = hidden * 3
            self.shared = nn.Sequential(
                nn.Linear(shared_in, hidden * 2),
                nn.LayerNorm(hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, seq_cat, seq_num, lengths, query_cat, query_num):
            row_x = torch.cat([self.row_cat(seq_cat), seq_num], dim=-1)
            row_x = self.row_proj(row_x)
            packed = nn.utils.rnn.pack_padded_sequence(
                row_x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h = self.gru(packed)
            h = h.view(2, 2, seq_cat.shape[0], -1)[-1]
            h = torch.cat([h[0], h[1]], dim=1)
            query_x = torch.cat([self.query_cat(query_cat), query_num], dim=-1)
            query_x = self.query_proj(query_x)
            return self.shared(torch.cat([h, query_x], dim=1))

    return SequenceBackbone()


def build_model(row_cards: List[int], query_cards: List[int], row_num_dim: int, query_num_dim: int, hidden: int, dropout: float):
    import torch.nn as nn

    class MultiTaskSequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = build_sequence_backbone(
                row_cards, query_cards, row_num_dim, query_num_dim, hidden, dropout
            )
            self.action_head = nn.Linear(hidden, ACTION_CLASSES)
            self.point_head = nn.Linear(hidden, POINT_CLASSES)
            self.server_head = nn.Linear(hidden, 1)

        def forward(self, seq_cat, seq_num, lengths, query_cat, query_num):
            z = self.backbone(seq_cat, seq_num, lengths, query_cat, query_num)
            return self.action_head(z), self.point_head(z), self.server_head(z).squeeze(1)

    return MultiTaskSequenceModel()


def build_single_task_model(
    row_cards: List[int],
    query_cards: List[int],
    row_num_dim: int,
    query_num_dim: int,
    hidden: int,
    dropout: float,
    task: str,
):
    import torch.nn as nn

    out_dim = {"action": ACTION_CLASSES, "point": POINT_CLASSES, "server": 1}[task]

    class SingleTaskSequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.task = task
            self.backbone = build_sequence_backbone(
                row_cards, query_cards, row_num_dim, query_num_dim, hidden, dropout
            )
            self.head = nn.Linear(hidden, out_dim)

        def forward(self, seq_cat, seq_num, lengths, query_cat, query_num):
            out = self.head(self.backbone(seq_cat, seq_num, lengths, query_cat, query_num))
            return out.squeeze(1) if self.task == "server" else out

    return SingleTaskSequenceModel()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for cls in range(n_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = ranks[y_true == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def score_predictions(meta: pd.DataFrame, action_prob: np.ndarray, point_prob: np.ndarray, server_prob: np.ndarray) -> Dict[str, float]:
    y_action = meta["target_action"].to_numpy(np.int64)
    y_point = meta["target_point"].to_numpy(np.int64)
    y_server = meta["target_server"].to_numpy(np.int64)
    s1 = macro_f1(y_action, action_prob.argmax(axis=1), ACTION_CLASSES)
    s2 = macro_f1(y_point, point_prob.argmax(axis=1), POINT_CLASSES)
    s3 = roc_auc(y_server, server_prob)
    return {"action_macro_f1": s1, "point_macro_f1": s2, "server_auc": s3, "overall": 0.4 * s1 + 0.4 * s2 + 0.2 * s3}


def select_device(args: argparse.Namespace):
    import torch

    has_cuda = torch.cuda.is_available()
    if args.require_gpu and (args.cpu or not has_cuda):
        raise RuntimeError(
            "GPU training was requested, but CUDA is not available in this runtime. "
            "Check the NVIDIA driver / container GPU passthrough, then rerun without --cpu."
        )
    return torch.device("cuda" if has_cuda and not args.cpu else "cpu")


def train_neural(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset | None,
    encoder: CategoryEncoder,
    args: argparse.Namespace,
    seed: int,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    seed_everything(seed)
    device = select_device(args)
    model = build_model(
        encoder.cardinalities(ROW_CAT_COLS),
        encoder.cardinalities(QUERY_CAT_COLS),
        len(ROW_NUM_COLS),
        len(QUERY_NUM_COLS),
        args.hidden,
        args.dropout,
    ).to(device)

    action_w = torch.as_tensor(class_weights(train_ds.y_action, ACTION_CLASSES, args.class_weight_power), device=device)
    point_w = torch.as_tensor(class_weights(train_ds.y_point, POINT_CLASSES, args.class_weight_power), device=device)
    pos = float(train_ds.y_server.sum())
    neg = float(len(train_ds.y_server) - pos)
    pos_weight = torch.as_tensor([neg / max(pos, 1.0)], device=device)
    action_loss = nn.CrossEntropyLoss(weight=action_w, label_smoothing=args.label_smoothing)
    point_loss = nn.CrossEntropyLoss(weight=point_w, label_smoothing=args.label_smoothing)
    server_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.08)

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate_fn(),
        pin_memory=(device.type == "cuda"),
    )
    best_state = None
    best_score = -1.0
    best_epoch = 0
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            seq_cat = batch["seq_cat"].to(device)
            seq_num = batch["seq_num"].to(device)
            lengths = batch["lengths"].to(device)
            query_cat = batch["query_cat"].to(device)
            query_num = batch["query_num"].to(device)
            y_action = batch["y_action"].to(device)
            y_point = batch["y_point"].to(device)
            y_server = batch["y_server"].to(device)
            out_action, out_point, out_server = model(seq_cat, seq_num, lengths, query_cat, query_num)
            loss = (
                args.action_loss_weight * action_loss(out_action, y_action)
                + args.point_loss_weight * point_loss(out_point, y_point)
                + args.server_loss_weight * server_loss(out_server, y_server)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(y_action)
        scheduler.step()
        train_loss = total_loss / len(train_ds)
        if val_ds is not None:
            action_prob, point_prob, server_prob = predict_neural(model, val_ds, args)
            metrics = score_predictions(val_ds.meta, action_prob, point_prob, server_prob)
            print(
                f"seed={seed} epoch={epoch:02d} loss={train_loss:.4f} "
                f"val_overall={metrics['overall']:.5f} "
                f"act={metrics['action_macro_f1']:.5f} point={metrics['point_macro_f1']:.5f} auc={metrics['server_auc']:.5f}",
                flush=True,
            )
            if metrics["overall"] > best_score:
                best_score = metrics["overall"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        else:
            print(f"seed={seed} epoch={epoch:02d} loss={train_loss:.4f}", flush=True)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"seed={seed} using epoch={best_epoch}", flush=True)
    return model


def train_single_task_neural(
    task: str,
    train_ds: SequenceDataset,
    val_ds: SequenceDataset | None,
    encoder: CategoryEncoder,
    args: argparse.Namespace,
    seed: int,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    seed_everything(seed)
    device = select_device(args)
    model = build_single_task_model(
        encoder.cardinalities(ROW_CAT_COLS),
        encoder.cardinalities(QUERY_CAT_COLS),
        len(ROW_NUM_COLS),
        len(QUERY_NUM_COLS),
        args.hidden,
        args.dropout,
        task,
    ).to(device)

    if task == "action":
        target_name = "y_action"
        y_train = train_ds.y_action
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights(y_train, ACTION_CLASSES, args.action_class_weight_power), device=device),
            label_smoothing=args.label_smoothing,
        )
    elif task == "point":
        target_name = "y_point"
        y_train = train_ds.y_point
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights(y_train, POINT_CLASSES, args.point_class_weight_power), device=device),
            label_smoothing=args.label_smoothing,
        )
    else:
        target_name = "y_server"
        y_train = train_ds.y_server
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([neg / max(pos, 1.0)], device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    task_epochs = {
        "action": args.action_epochs or args.epochs,
        "point": args.point_epochs or args.epochs,
        "server": args.server_epochs or args.epochs,
    }[task]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(task_epochs, 1), eta_min=args.lr * 0.08)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate_fn(),
        pin_memory=(device.type == "cuda"),
    )

    best_state = None
    best_score = -1.0
    best_epoch = 0
    patience_left = args.patience
    for epoch in range(1, task_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["seq_cat"].to(device),
                batch["seq_num"].to(device),
                batch["lengths"].to(device),
                batch["query_cat"].to(device),
                batch["query_num"].to(device),
            )
            y = batch[target_name].to(device)
            loss = criterion(logits, y if task != "server" else y.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
        scheduler.step()
        train_loss = total_loss / len(train_ds)
        if val_ds is not None:
            pred = predict_single_task_neural(model, val_ds, args, task)
            if task == "action":
                score = macro_f1(val_ds.y_action, pred.argmax(axis=1), ACTION_CLASSES)
                metric_name = "action_macro_f1"
            elif task == "point":
                score = macro_f1(val_ds.y_point, pred.argmax(axis=1), POINT_CLASSES)
                metric_name = "point_macro_f1"
            else:
                score = roc_auc(val_ds.y_server, pred)
                metric_name = "server_auc"
            print(
                f"task={task} seed={seed} epoch={epoch:02d} loss={train_loss:.4f} "
                f"val_{metric_name}={score:.5f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        else:
            print(f"task={task} seed={seed} epoch={epoch:02d} loss={train_loss:.4f}", flush=True)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"task={task} seed={seed} using epoch={best_epoch}", flush=True)
    return model


def predict_neural(model, dataset: SequenceDataset, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate_fn(),
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    action_parts = []
    point_parts = []
    server_parts = []
    with torch.no_grad():
        for batch in loader:
            out_action, out_point, out_server = model(
                batch["seq_cat"].to(device),
                batch["seq_num"].to(device),
                batch["lengths"].to(device),
                batch["query_cat"].to(device),
                batch["query_num"].to(device),
            )
            action_parts.append(torch.softmax(out_action, dim=1).cpu().numpy())
            point_parts.append(torch.softmax(out_point, dim=1).cpu().numpy())
            server_parts.append(torch.sigmoid(out_server).cpu().numpy())
    return np.vstack(action_parts), np.vstack(point_parts), np.concatenate(server_parts)


def predict_single_task_neural(model, dataset: SequenceDataset, args: argparse.Namespace, task: str) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate_fn(),
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["seq_cat"].to(device),
                batch["seq_num"].to(device),
                batch["lengths"].to(device),
                batch["query_cat"].to(device),
                batch["query_num"].to(device),
            )
            if task == "server":
                parts.append(torch.sigmoid(out).cpu().numpy())
            else:
                parts.append(torch.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(parts) if task == "server" else np.vstack(parts)


def train_single_task_ensemble(
    train_ds: SequenceDataset,
    pred_ds: SequenceDataset,
    encoder: CategoryEncoder,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = {"action": [], "point": [], "server": []}
    for task in ["action", "point", "server"]:
        for seed in args.seeds:
            model = train_single_task_neural(task, train_ds, None, encoder, args, seed)
            preds[task].append(predict_single_task_neural(model, pred_ds, args, task))
    return (
        np.mean(preds["action"], axis=0),
        np.mean(preds["point"], axis=0),
        np.mean(preds["server"], axis=0),
    )


class BackoffMulticlass:
    def __init__(self, n_classes: int, key_specs):
        self.n_classes = n_classes
        self.key_specs = key_specs
        self.prior = np.ones(n_classes, dtype=np.float64) / n_classes
        self.tables = []

    @staticmethod
    def _key(row, keys):
        return tuple(int(row[k]) for k in keys)

    def fit(self, frame: pd.DataFrame, target_col: str):
        counts = np.bincount(frame[target_col].to_numpy(np.int64), minlength=self.n_classes).astype(np.float64)
        self.prior = (counts + 1.0) / (counts.sum() + self.n_classes)
        self.tables = []
        for keys, weight, smooth in self.key_specs:
            table = {}
            for values, group in frame.groupby(list(keys), sort=False):
                if not isinstance(values, tuple):
                    values = (values,)
                y = group[target_col].to_numpy(np.int64)
                table[tuple(int(v) for v in values)] = np.bincount(y, minlength=self.n_classes).astype(np.float64)
            self.tables.append((keys, float(weight), float(smooth), table))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(frame), self.n_classes), dtype=np.float64)
        for i, (_, row) in enumerate(frame.iterrows()):
            acc = 0.25 * self.prior.copy()
            total = 0.25
            for keys, weight, smooth, table in self.tables:
                counts = table.get(self._key(row, keys))
                if counts is None:
                    continue
                n = counts.sum()
                reliability = n / (n + smooth)
                dist = (counts + self.prior) / (n + 1.0)
                w = weight * reliability
                acc += w * dist
                total += w
            out[i] = acc / total
        return out.astype(np.float32)


class BackoffBinary:
    def __init__(self, key_specs):
        self.key_specs = key_specs
        self.prior = 0.5
        self.tables = []

    @staticmethod
    def _key(row, keys):
        return tuple(int(row[k]) for k in keys)

    def fit(self, frame: pd.DataFrame, target_col: str):
        y = frame[target_col].to_numpy(np.float64)
        self.prior = float((y.sum() + 1.0) / (len(y) + 2.0))
        self.tables = []
        for keys, weight, smooth in self.key_specs:
            table = {}
            for values, group in frame.groupby(list(keys), sort=False):
                if not isinstance(values, tuple):
                    values = (values,)
                yy = group[target_col].to_numpy(np.float64)
                table[tuple(int(v) for v in values)] = (float(yy.sum()), float(len(yy)))
            self.tables.append((keys, float(weight), float(smooth), table))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(frame), dtype=np.float64)
        for i, (_, row) in enumerate(frame.iterrows()):
            acc = 0.25 * self.prior
            total = 0.25
            for keys, weight, smooth, table in self.tables:
                val = table.get(self._key(row, keys))
                if val is None:
                    continue
                pos, n = val
                reliability = n / (n + smooth)
                prob = (pos + self.prior) / (n + 1.0)
                w = weight * reliability
                acc += w * prob
                total += w
            out[i] = acc / total
        return out.astype(np.float32)


def fit_backoff(meta: pd.DataFrame):
    action = BackoffMulticlass(ACTION_CLASSES, ACTION_KEY_SPECS).fit(meta, "target_action")
    point = BackoffMulticlass(POINT_CLASSES, POINT_KEY_SPECS).fit(meta, "target_point")
    server = BackoffBinary(SERVER_KEY_SPECS).fit(meta, "target_server")
    return action, point, server


def fit_split_backoff(action_point_meta: pd.DataFrame, server_meta: pd.DataFrame):
    action = BackoffMulticlass(ACTION_CLASSES, ACTION_KEY_SPECS).fit(action_point_meta, "target_action")
    point = BackoffMulticlass(POINT_CLASSES, POINT_KEY_SPECS).fit(action_point_meta, "target_point")
    server = BackoffBinary(SERVER_KEY_SPECS).fit(server_meta, "target_server")
    return action, point, server


def predict_backoff(models, meta: pd.DataFrame):
    action, point, server = models
    return action.predict_proba(meta), point.predict_proba(meta), server.predict_proba(meta)


def split_validation_meta(meta: pd.DataFrame, val_match_min: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    train_only = meta[meta["source"] == "train"]
    if val_match_min is None:
        matches = sorted(int(x) for x in train_only["match"].unique())
        val_count = max(1, int(round(len(matches) * 0.18)))
        val_matches = set(matches[-val_count:])
    else:
        val_matches = set(int(x) for x in train_only.loc[train_only["match"] >= val_match_min, "match"].unique())
    val_mask = (meta["source"] == "train") & meta["match"].isin(val_matches)
    train_mask = ~meta["match"].isin(val_matches)
    return meta[train_mask].reset_index(drop=True), meta[val_mask].reset_index(drop=True), sorted(val_matches)


def blend_probs(nn_prob, backoff_prob, nn_weight: float):
    return nn_weight * nn_prob + (1.0 - nn_weight) * backoff_prob


def prior_adjust_probs(prob: np.ndarray, train_y: np.ndarray, n_classes: int, strength: float) -> np.ndarray:
    if strength <= 0:
        return prob
    counts = np.bincount(train_y.astype(np.int64), minlength=n_classes).astype(np.float64) + 1.0
    prior = counts / counts.sum()
    adjusted = prob / np.power(prior[None, :], strength)
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    return adjusted.astype(np.float32)


def apply_multiclass_adjustments(
    train_meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    action_strength: float,
    point_strength: float,
) -> Tuple[np.ndarray, np.ndarray]:
    action_prob = prior_adjust_probs(
        action_prob,
        train_meta["target_action"].to_numpy(np.int64),
        ACTION_CLASSES,
        action_strength,
    )
    point_prob = prior_adjust_probs(
        point_prob,
        train_meta["target_point"].to_numpy(np.int64),
        POINT_CLASSES,
        point_strength,
    )
    return action_prob, point_prob


def optimize_blend(val_meta, train_meta, nn_preds, backoff_preds):
    best_action = None
    for aw in np.linspace(0.45, 0.95, 11):
        base_action = blend_probs(nn_preds[0], backoff_preds[0], float(aw))
        for action_adj in np.linspace(0.0, 1.0, 21):
            action_prob = prior_adjust_probs(
                base_action,
                train_meta["target_action"].to_numpy(np.int64),
                ACTION_CLASSES,
                float(action_adj),
            )
            score = macro_f1(val_meta["target_action"].to_numpy(np.int64), action_prob.argmax(axis=1), ACTION_CLASSES)
            record = (score, float(aw), float(action_adj))
            if best_action is None or record[0] > best_action[0]:
                best_action = record

    best_point = None
    for pw in np.linspace(0.45, 0.95, 11):
        base_point = blend_probs(nn_preds[1], backoff_preds[1], float(pw))
        for point_adj in np.linspace(0.0, 1.0, 21):
            point_prob = prior_adjust_probs(
                base_point,
                train_meta["target_point"].to_numpy(np.int64),
                POINT_CLASSES,
                float(point_adj),
            )
            score = macro_f1(val_meta["target_point"].to_numpy(np.int64), point_prob.argmax(axis=1), POINT_CLASSES)
            record = (score, float(pw), float(point_adj))
            if best_point is None or record[0] > best_point[0]:
                best_point = record

    best_server = None
    for sw in np.linspace(0.35, 0.95, 13):
        server_prob = blend_probs(nn_preds[2], backoff_preds[2], float(sw))
        score = roc_auc(val_meta["target_server"].to_numpy(np.int64), server_prob)
        record = (score, float(sw))
        if best_server is None or record[0] > best_server[0]:
            best_server = record

    metrics = {
        "action_macro_f1": best_action[0],
        "point_macro_f1": best_point[0],
        "server_auc": best_server[0],
        "overall": 0.4 * best_action[0] + 0.4 * best_point[0] + 0.2 * best_server[0],
    }
    return (
        metrics["overall"],
        best_action[1],
        best_point[1],
        best_server[1],
        best_action[2],
        best_point[2],
        metrics,
    )


def read_inputs(data_dir: Path):
    train = add_derived(pd.read_csv(data_dir / "train.csv"), has_server=True)
    old = add_derived(pd.read_csv(data_dir / "test_old.csv"), has_server=True)
    new = add_derived(pd.read_csv(data_dir / "test_new.csv"), has_server=False)
    return train, old, new


def train_and_predict(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    if args.require_gpu and (args.validate or args.method in {"neural", "ensemble"}):
        select_device(args)
    train, old, new = read_inputs(data_dir)

    train_meta = build_transition_meta(train, "train")
    old_meta = build_transition_meta(old, "old")
    all_meta = pd.concat([train_meta, old_meta], ignore_index=True)
    observed_test_meta = build_observed_transition_meta(new, "test_new_observed")
    if args.use_test_observed_transitions:
        action_point_meta = pd.concat([all_meta, observed_test_meta], ignore_index=True)
        action_point_strokes = pd.concat([train, old, new], ignore_index=True)
    else:
        action_point_meta = all_meta
        action_point_strokes = pd.concat([train, old], ignore_index=True)
    test_meta = build_test_meta(new)
    print(f"transition examples: train={len(train_meta)} old={len(old_meta)} all={len(all_meta)}", flush=True)
    print(
        f"observed test transitions for action/point: {len(observed_test_meta)} "
        f"used={args.use_test_observed_transitions}",
        flush=True,
    )
    print(f"test rallies: {len(test_meta)}", flush=True)

    if args.validate:
        tr_meta, val_meta, val_matches = split_validation_meta(all_meta, args.val_match_min)
        print(f"validation matches: {val_matches[:5]} ... {val_matches[-5:]} ({len(val_matches)} matches)", flush=True)
        tr_strokes = pd.concat(
            [
                train[train["rally_uid"].isin(tr_meta.loc[tr_meta["source"] == "train", "rally_uid"].unique())],
                old[old["rally_uid"].isin(tr_meta.loc[tr_meta["source"] == "old", "rally_uid"].unique())],
            ],
            ignore_index=True,
        )
        val_strokes = train[train["rally_uid"].isin(val_meta["rally_uid"].unique())].reset_index(drop=True)
        encoder = CategoryEncoder(sorted(set(ROW_CAT_COLS + QUERY_CAT_COLS))).fit([tr_strokes, tr_meta])
        tr_ds = SequenceDataset(tr_strokes, tr_meta, encoder)
        val_ds = SequenceDataset(val_strokes, val_meta, encoder)
        backoff = fit_backoff(tr_meta)
        backoff_val = predict_backoff(backoff, val_meta)
        backoff_metrics = score_predictions(val_meta, *backoff_val)
        print("backoff validation:", json.dumps(backoff_metrics, ensure_ascii=False, indent=2), flush=True)
        if args.model_version == "multitask":
            nn_val_parts = []
            for seed in args.seeds:
                model = train_neural(tr_ds, val_ds, encoder, args, seed)
                nn_val_parts.append(predict_neural(model, val_ds, args))
            nn_val = tuple(np.mean([part[i] for part in nn_val_parts], axis=0) for i in range(3))
        else:
            task_parts = {"action": [], "point": [], "server": []}
            for task in ["action", "point", "server"]:
                for seed in args.seeds:
                    model = train_single_task_neural(task, tr_ds, val_ds, encoder, args, seed)
                    task_parts[task].append(predict_single_task_neural(model, val_ds, args, task))
            nn_val = (
                np.mean(task_parts["action"], axis=0),
                np.mean(task_parts["point"], axis=0),
                np.mean(task_parts["server"], axis=0),
            )
        nn_metrics = score_predictions(val_meta, *nn_val)
        print("neural validation:", json.dumps(nn_metrics, ensure_ascii=False, indent=2), flush=True)
        best = optimize_blend(val_meta, tr_meta, nn_val, backoff_val)
        print(
            "best blend validation:",
            json.dumps(
                {
                    "overall": best[0],
                    "action_nn_weight": best[1],
                    "point_nn_weight": best[2],
                    "server_nn_weight": best[3],
                    "action_prior_adjust": best[4],
                    "point_prior_adjust": best[5],
                    "metrics": best[6],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    stroke_all = pd.concat([train, old], ignore_index=True)
    encoder = CategoryEncoder(sorted(set(ROW_CAT_COLS + QUERY_CAT_COLS))).fit([stroke_all, all_meta])
    train_ds = SequenceDataset(stroke_all, all_meta, encoder)
    test_ds = SequenceDataset(new, test_meta, encoder)

    ap_encoder = CategoryEncoder(sorted(set(ROW_CAT_COLS + QUERY_CAT_COLS))).fit(
        [action_point_strokes, action_point_meta, test_meta]
    )
    ap_train_ds = SequenceDataset(action_point_strokes, action_point_meta, ap_encoder)
    ap_test_ds = SequenceDataset(new, test_meta, ap_encoder)
    server_train_ds = train_ds
    server_test_ds = test_ds

    backoff = fit_split_backoff(action_point_meta, all_meta)
    backoff_test = predict_backoff(backoff, test_meta)

    nn_test_parts = []
    if args.method in {"neural", "ensemble"}:
        if args.model_version == "multitask":
            for seed in args.seeds:
                model = train_neural(train_ds, None, encoder, args, seed)
                nn_test_parts.append(predict_neural(model, test_ds, args))
            nn_test = tuple(np.mean([part[i] for part in nn_test_parts], axis=0) for i in range(3))
        else:
            task_parts = {"action": [], "point": [], "server": []}
            for seed in args.seeds:
                model = train_single_task_neural("action", ap_train_ds, None, ap_encoder, args, seed)
                task_parts["action"].append(predict_single_task_neural(model, ap_test_ds, args, "action"))
            for seed in args.seeds:
                model = train_single_task_neural("point", ap_train_ds, None, ap_encoder, args, seed)
                task_parts["point"].append(predict_single_task_neural(model, ap_test_ds, args, "point"))
            for seed in args.seeds:
                model = train_single_task_neural("server", server_train_ds, None, encoder, args, seed)
                task_parts["server"].append(predict_single_task_neural(model, server_test_ds, args, "server"))
            nn_test = (
                np.mean(task_parts["action"], axis=0),
                np.mean(task_parts["point"], axis=0),
                np.mean(task_parts["server"], axis=0),
            )
    else:
        nn_test = backoff_test

    if args.method == "backoff":
        action_prob, point_prob, server_prob = backoff_test
    elif args.method == "neural":
        action_prob, point_prob, server_prob = nn_test
    else:
        action_prob = blend_probs(nn_test[0], backoff_test[0], args.action_nn_weight)
        point_prob = blend_probs(nn_test[1], backoff_test[1], args.point_nn_weight)
        server_prob = blend_probs(nn_test[2], backoff_test[2], args.server_nn_weight)

    action_prob, point_prob = apply_multiclass_adjustments(
        action_point_meta,
        action_prob,
        point_prob,
        args.action_prior_adjust,
        args.point_prior_adjust,
    )

    submission = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_prob.argmax(axis=1).astype(int),
            "pointId": point_prob.argmax(axis=1).astype(int),
            "serverGetPoint": server_prob.astype(float),
        }
    )

    submission.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {out_path} rows={len(submission)}", flush=True)
    print(submission.head(10).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output", default="submission_v3.csv")
    parser.add_argument("--method", choices=["backoff", "neural", "ensemble"], default="ensemble")
    parser.add_argument("--model-version", choices=["single-task", "multitask"], default="single-task")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--val-match-min", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 777])
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--action-epochs", type=int, default=None)
    parser.add_argument("--point-epochs", type=int, default=None)
    parser.add_argument("--server-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=2.5)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--class-weight-power", type=float, default=0.45)
    parser.add_argument("--action-class-weight-power", type=float, default=0.50)
    parser.add_argument("--point-class-weight-power", type=float, default=0.50)
    parser.add_argument("--action-loss-weight", type=float, default=0.45)
    parser.add_argument("--point-loss-weight", type=float, default=0.45)
    parser.add_argument("--server-loss-weight", type=float, default=0.20)
    parser.add_argument("--action-nn-weight", type=float, default=0.45)
    parser.add_argument("--point-nn-weight", type=float, default=0.45)
    parser.add_argument("--server-nn-weight", type=float, default=0.35)
    parser.add_argument("--action-prior-adjust", type=float, default=0.35)
    parser.add_argument("--point-prior-adjust", type=float, default=0.40)
    parser.add_argument("--use-test-observed-transitions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_and_predict(parse_args())
