import logging
from typing import Iterable, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_distribution(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        raise ValueError("Distribution cannot sum to zero")
    return values / total


def compute_shannon_entropy(sequence: Iterable[str]) -> float:
    values, counts = np.unique(list(sequence), return_counts=True)
    probabilities = counts / counts.sum()
    return -float(np.sum(probabilities * np.log2(probabilities + 1e-12)))


def compute_cross_entropy(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    p_values = np.array(list(p.values()), dtype=float)
    q_values = np.array(list(q.values()), dtype=float)
    q_norm = _normalize_distribution(q_values)
    return -float(np.sum(p_values * np.log2(q_norm + 1e-12)))


def compute_kl_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    p_values = _normalize_distribution(np.array(list(p.values()), dtype=float))
    q_values = _normalize_distribution(np.array(list(q.values()), dtype=float))
    return float(np.sum(p_values * np.log2((p_values + 1e-12) / (q_values + 1e-12))))


def compute_js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    p_values = _normalize_distribution(np.array(list(p.values()), dtype=float))
    q_values = _normalize_distribution(np.array(list(q.values()), dtype=float))
    m = 0.5 * (p_values + q_values)
    return 0.5 * (compute_kl_divergence(dict(enumerate(p_values)), dict(enumerate(m))) + compute_kl_divergence(dict(enumerate(q_values)), dict(enumerate(m))))


def compute_mutual_information(x: Iterable[str], y: Iterable[str]) -> float:
    x_list = list(x)
    y_list = list(y)
    joint = {}
    px = {}
    py = {}
    total = len(x_list)
    for xi, yi in zip(x_list, y_list):
        joint[(xi, yi)] = joint.get((xi, yi), 0) + 1
        px[xi] = px.get(xi, 0) + 1
        py[yi] = py.get(yi, 0) + 1
    mi = 0.0
    for (xi, yi), count in joint.items():
        pxy = count / total
        mi += pxy * np.log2(pxy / ((px[xi] / total) * (py[yi] / total) + 1e-12) + 1e-12)
    return float(max(mi, 0.0))


def compute_conditional_entropy(x: Iterable[str], y: Iterable[str]) -> float:
    mi = compute_mutual_information(x, y)
    entropy_x = compute_shannon_entropy(x)
    return max(0.0, entropy_x - mi)
