"""Text-domain adversarial attacks against a fine-tuned classifier.

Adapts L-inf intuitions to discrete text. None of these are sound under
gradient masking — they're synonym/typo perturbations that work well
against classifiers without robust training.

Implements:
- TextFooler-style synonym swap (Jin et al. 2020)
- Random character noise (insertion / deletion / swap)
- Importance-ranked word deletion

The point is to compare a clean model vs. one trained with defensive
distillation: distillation reduces gradient magnitude but does NOT
actually defend against these decision-boundary-finding attacks.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class AttackResult:
    original_text: str
    adversarial_text: str
    queries: int
    succeeded: bool
    original_pred: int
    final_pred: int


def char_noise(text: str, n_perturbations: int = 3, seed: int = 0) -> str:
    """Random character-level noise: insertion / deletion / swap."""
    rng = random.Random(seed)
    chars = list(text)
    for _ in range(n_perturbations):
        if not chars:
            break
        op = rng.choice(['insert', 'delete', 'swap'])
        i = rng.randrange(len(chars))
        if op == 'insert':
            chars.insert(i, rng.choice(string.ascii_lowercase))
        elif op == 'delete':
            del chars[i]
        elif op == 'swap' and i + 1 < len(chars):
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return ''.join(chars)


