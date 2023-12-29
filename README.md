# 2020 — GPT-2 Generation + Defensive Distillation

Two threads of 2020 in one project: open-ended text generation with GPT-2, and
defensive distillation as a (now-discredited) defense against adversarial examples.

## What's here

- `src/gpt2_generate.py` — load GPT-2 medium, sample with top-k / top-p
- `src/defensive_distill.py` — train a teacher, then distill into a student at high
  softmax temperature; show how it appears to defend against FGSM but breaks under PGD
  with the right loss formulation

## Run

```bash
pip install -r requirements.txt
python src/gpt2_generate.py --prompt "The most important security risk of LLMs is"
python src/defensive_distill.py
```

