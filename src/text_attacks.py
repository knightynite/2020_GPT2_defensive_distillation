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


def word_importance(
    predict_proba: Callable[[str], List[float]],
    text: str,
    target_class: int,
) -> List[float]:
    """For each word, score importance as drop in target-class prob when removed."""
    words = text.split()
    base = predict_proba(text)[target_class]
    scores = []
    for i in range(len(words)):
        masked = ' '.join(words[:i] + words[i+1:])
        new = predict_proba(masked)[target_class]
        scores.append(base - new)
    return scores


def importance_ranked_deletion(
    predict_proba: Callable[[str], List[float]],
    text: str,
    original_pred: int,
    max_deletions: int = 5,
) -> AttackResult:
    """Greedy attack: delete words in importance order until prediction flips."""
    words = text.split()
    queries = 0
    scores = word_importance(predict_proba, text, original_pred)
    queries += len(words) + 1
    order = sorted(range(len(words)), key=lambda i: -scores[i])

    current_text = text
    for k in range(1, max_deletions + 1):
        keep = sorted(order[k:])
        candidate = ' '.join(words[i] for i in keep)
        probs = predict_proba(candidate)
        queries += 1
        new_pred = max(range(len(probs)), key=lambda j: probs[j])
        if new_pred != original_pred:
            return AttackResult(
                original_text=text,
                adversarial_text=candidate,
                queries=queries,
                succeeded=True,
                original_pred=original_pred,
                final_pred=new_pred,
            )
    return AttackResult(
        original_text=text,
        adversarial_text=current_text,
        queries=queries,
        succeeded=False,
        original_pred=original_pred,
        final_pred=original_pred,
    )


def synonym_swap_attack(
    predict_proba: Callable[[str], List[float]],
    text: str,
    original_pred: int,
    synonym_dict: dict,
    max_swaps: int = 5,
) -> AttackResult:
    """TextFooler-style synonym replacement.

    `synonym_dict` maps each candidate word to a list of synonyms. In
    practice you'd use counter-fitted GloVe + a sense filter; here we
    parameterize so callers can plug in whatever vocabulary they want.
    """
    words = text.split()
    queries = 0
    base_scores = word_importance(predict_proba, text, original_pred)
    queries += len(words) + 1
    ranked = sorted(range(len(words)), key=lambda i: -base_scores[i])

    current = list(words)
    for idx in ranked[:max_swaps]:
        word = current[idx].lower().strip(string.punctuation)
        candidates = synonym_dict.get(word, [])
        best_drop = 0.0
        best_word = None
        for cand in candidates:
            trial = list(current)
            trial[idx] = cand
            probs = predict_proba(' '.join(trial))
            queries += 1
            drop = base_scores[idx] - (probs[original_pred] - 0.0)
            if drop > best_drop:
                best_drop = drop
                best_word = cand
        if best_word:
            current[idx] = best_word
            probs = predict_proba(' '.join(current))
            queries += 1
            new_pred = max(range(len(probs)), key=lambda j: probs[j])
            if new_pred != original_pred:
                return AttackResult(
                    original_text=text,
                    adversarial_text=' '.join(current),
                    queries=queries,
                    succeeded=True,
                    original_pred=original_pred,
                    final_pred=new_pred,
                )
    return AttackResult(
        original_text=text,
        adversarial_text=' '.join(current),
        queries=queries,
        succeeded=False,
        original_pred=original_pred,
        final_pred=original_pred,
    )
