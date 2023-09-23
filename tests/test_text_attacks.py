"""Tests for text-domain attacks. Uses a deterministic mock predictor so
we can verify attack mechanics without loading a transformer."""
import unittest

from src.text_attacks import (
    AttackResult,
    char_noise,
    importance_ranked_deletion,
    synonym_swap_attack,
    word_importance,
)


def make_keyword_predictor(target_words):
    """Returns a predict_proba(text) that flips on the presence of any
    target word. Class 0 = absent, class 1 = present.
    """
    target_set = set(target_words)
    def predict(text):
        words = set(text.lower().split())
        if words & target_set:
            return [0.05, 0.95]
        return [0.95, 0.05]
    return predict


class TestCharNoise(unittest.TestCase):
    def test_changes_text(self):
        out = char_noise("hello world this is fine", n_perturbations=5, seed=0)
        self.assertNotEqual(out, "hello world this is fine")

    def test_deterministic_with_seed(self):
        a = char_noise("hello world", n_perturbations=3, seed=42)
        b = char_noise("hello world", n_perturbations=3, seed=42)
        self.assertEqual(a, b)


class TestWordImportance(unittest.TestCase):
    def test_target_word_is_most_important(self):
        predict = make_keyword_predictor(['phishing'])
        scores = word_importance(predict, "this is a phishing example email", 1)
        # Removing "phishing" should drop class-1 prob the most
        words = "this is a phishing example email".split()
        idx = words.index('phishing')
        self.assertGreater(scores[idx], max(s for i, s in enumerate(scores) if i != idx))


class TestImportanceRankedDeletion(unittest.TestCase):
    def test_succeeds_when_target_word_present(self):
        predict = make_keyword_predictor(['phishing'])
        result = importance_ranked_deletion(
            predict, "alert: phishing detected here", 1, max_deletions=2
        )
        self.assertTrue(result.succeeded)
        self.assertNotIn('phishing', result.adversarial_text.lower())

    def test_fails_gracefully(self):
        predict = make_keyword_predictor(['phishing'])
        # Original prediction is class 0 (no target word); attack to flip
        # to class 1 by deleting words is impossible.
        result = importance_ranked_deletion(
            predict, "totally benign innocuous text", 0, max_deletions=3
        )
        self.assertFalse(result.succeeded)


class TestSynonymSwap(unittest.TestCase):
    def test_swaps_when_synonym_drops_score(self):
        predict = make_keyword_predictor(['great'])
        synonyms = {'great': ['mediocre']}  # "mediocre" not in target_set
        result = synonym_swap_attack(
            predict, "today was a great day", 1, synonyms, max_swaps=1
        )
        self.assertTrue(result.succeeded)
        self.assertIn('mediocre', result.adversarial_text)


if __name__ == '__main__':
    unittest.main()
