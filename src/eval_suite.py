"""Evaluation harness comparing clean vs. defensively-distilled classifiers.

The setup uses a simple sentiment / toxicity classifier (loaded from a
HuggingFace checkpoint) as the "victim". We then run text attacks against
both the victim and a distilled student, and report:

  - clean accuracy (sanity check distillation didn't tank utility)
  - attack success rate per attack type
  - mean queries to flip
  - the gradient-masking diagnostic: ratio of FGSM-style probability
    distance vs. importance-ranked deletion success — these should
    track each other; if FGSM looks defended but ranked-deletion still
    flips, you have masking, not robustness.
"""
import argparse
import json
import os
from dataclasses import asdict
from statistics import mean

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.text_attacks import (
    AttackResult,
    char_noise,
    importance_ranked_deletion,
    synonym_swap_attack,
)


SIMPLE_SYNONYMS = {
    'good': ['fine', 'decent', 'okay', 'pleasant'],
    'bad': ['poor', 'awful', 'terrible', 'unfortunate'],
    'great': ['excellent', 'fantastic', 'wonderful'],
    'terrible': ['horrible', 'dreadful', 'atrocious'],
    'love': ['adore', 'enjoy', 'appreciate'],
    'hate': ['despise', 'loathe', 'detest'],
    'best': ['finest', 'top', 'greatest'],
    'worst': ['poorest', 'lousiest'],
}


def make_predict(model, tokenizer, device, max_len=128):
    def predict_proba(text):
        enc = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=max_len,
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)[0].cpu().tolist()
        return probs
    return predict_proba


def evaluate(model_name: str, examples: list, max_attacks: int = 50) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    predict = make_predict(model, tokenizer, device)

    n = 0
    n_correct = 0
    delete_results = []
    synonym_results = []
    char_flips = 0
    for text, label in examples[:max_attacks]:
        probs = predict(text)
        pred = max(range(len(probs)), key=lambda i: probs[i])
        n += 1
        if pred == label:
            n_correct += 1
        else:
            continue  # only attack examples the model already gets right

        delete_results.append(
            importance_ranked_deletion(predict, text, pred)
        )
        synonym_results.append(
            synonym_swap_attack(predict, text, pred, SIMPLE_SYNONYMS)
        )
        noisy = char_noise(text, n_perturbations=5, seed=hash(text) & 0xFFFF)
        np_probs = predict(noisy)
        if max(range(len(np_probs)), key=lambda i: np_probs[i]) != pred:
            char_flips += 1

    return {
        'n_examples': n,
        'clean_accuracy': n_correct / n if n else float('nan'),
        'deletion_success_rate':
            sum(r.succeeded for r in delete_results) / max(len(delete_results), 1),
        'deletion_mean_queries':
            mean([r.queries for r in delete_results]) if delete_results else 0,
        'synonym_success_rate':
            sum(r.succeeded for r in synonym_results) / max(len(synonym_results), 1),
        'char_noise_flip_rate':
            char_flips / max(n_correct, 1),
        'sample_results': {
            'deletion': [asdict(r) for r in delete_results[:3]],
            'synonym': [asdict(r) for r in synonym_results[:3]],
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clean-model', required=True,
                   help='HF checkpoint of the clean classifier')
    p.add_argument('--distilled-model', required=True,
                   help='HF checkpoint of the distilled student')
    p.add_argument('--examples-json', required=True,
                   help='JSON list of [text, label] pairs')
    p.add_argument('--out', default='results/eval_suite.json')
    p.add_argument('--max-attacks', type=int, default=50)
    args = p.parse_args()

    with open(args.examples_json) as f:
        examples = json.load(f)

    clean = evaluate(args.clean_model, examples, args.max_attacks)
    distilled = evaluate(args.distilled_model, examples, args.max_attacks)

    summary = {
        'clean': clean,
        'distilled': distilled,
        'gradient_masking_diagnostic':
            ('Char-noise flips dropped' if distilled['char_noise_flip_rate'] <
                clean['char_noise_flip_rate']
             else 'Char-noise unchanged') +
            ' BUT ' +
            ('deletion success similar' if abs(distilled['deletion_success_rate'] -
                clean['deletion_success_rate']) < 0.10
             else 'deletion success dropped'),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
