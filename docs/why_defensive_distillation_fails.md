# Why defensive distillation fails

Defensive distillation (Papernot et al. 2016) trains a student network on
softmax outputs of a teacher trained at high temperature. The pitch is
that the student inherits the teacher's decision boundaries with a much
"smoother" loss landscape, which should reduce sensitivity to small
perturbations.

It doesn't. Here's why, and what the failure pattern looks like in
practice.

## The intended mechanism

For input x and one-hot label y, cross-entropy loss is:
  L(x, y) = -log softmax_y(z(x)/T)

Where z(x) is logits and T is temperature. The gradient w.r.t. x is
proportional to 1/T (since divisions inside the softmax compound). So at
T=100, gradients are 100× smaller than at T=1. FGSM-style attacks
(`x + ε · sign(grad)`) therefore appear to fail: the gradient direction is
noisy and the magnitude is small.

## Why it doesn't work

The classifier's *decision boundaries* didn't move. They were inherited
verbatim from the teacher. What changed is the *gradient* at points
*near but not on* the boundaries — and only because of a constant scaling
factor.

Two attacks bypass this completely:

1. **Carlini-Wagner with logit-difference loss.** Instead of the
   cross-entropy loss whose gradient is shrunk by 1/T, use:
     f(x) = max(z_y(x) - max_{j ≠ y} z_j(x), -κ)
   This loss is computed on raw logits, not softmax. The 1/T scaling
   never enters. The minimum-perturbation attack works as well as it
   did against the undefended model.

2. **Decision-boundary attacks (DeepFool, Boundary Attack).** These
   don't use gradients of cross-entropy at all. They use the *geometry*
   of the decision surface, which is unchanged.

## The diagnostic

If your "defense" makes FGSM accuracy go up but doesn't change CW or
DeepFool numbers, you have **gradient masking**, not robustness. The
canonical reference is Athalye, Carlini, Wagner 2018, *Obfuscated
Gradients Give a False Sense of Security*. They show that 7 of 9 then-
recent defenses suffered from variations of this problem.

Rules of thumb for spotting masking:

- FGSM accuracy > PGD accuracy by a wide margin: suspicious. PGD with
  enough iterations and random restarts should match or exceed FGSM in
  attack success.
- White-box accuracy > black-box accuracy: very suspicious. White-box
  is strictly more powerful than black-box.
- Iterative attacks plateau early: the optimization is being misled by
  obfuscated gradients.
- Increasing ε doesn't decrease robust accuracy linearly: usually means
  the attack is failing for a non-robustness reason.

## What works instead

Adversarial training (Madry et al. 2018). At each minibatch, run an
inner-loop PGD attack to construct adversarial examples; train the model
on those instead of clean ones. This actually changes the loss landscape
because the network has to fit the adversarially-perturbed distribution.

Adversarial training has its own caveats — overfits to the training ε,
brittle to threat-model shifts, slow to train — but it's the only first-
order defense that has held up over time.

## Pedagogical value

Why include defensive distillation in this repository at all if it's
broken?

Because it's a *clean* example of how easy it is to convince yourself
you've defended a model when you haven't. Reading the original Papernot
2016 paper, the intuition is appealing. Implementing it produces
plausible-looking numbers. Only when you bring out the right attack
do you see it was masking gradients all along.

That experience is valuable for anyone working on ML security. The next
defense you're tempted to publish should clear that bar before you
trust it.

## References

- Papernot et al. 2016 — *Distillation as a Defense to Adversarial
  Perturbations against Deep Neural Networks*
- Carlini & Wagner 2017 — *Towards Evaluating the Robustness of Neural
  Networks* (broke distillation explicitly)
- Athalye, Carlini, Wagner 2018 — *Obfuscated Gradients Give a False
  Sense of Security* (broader context)
- Madry et al. 2018 — *Towards Deep Learning Models Resistant to
  Adversarial Attacks* (adversarial training)
- Tramèr et al. 2020 — *On Adaptive Attacks to Adversarial Example
  Defenses*
