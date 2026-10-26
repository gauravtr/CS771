# PUF Modeling and Inversion — honest results

Numbers below are from an actual run of `main.py` (seed=42, 20,000 CRPs,
64-bit challenges, k=4 sub-PUFs for the XOR case). Re-run it yourself —
your exact numbers will vary slightly with the random seed, but the
pattern will hold.

## 1. Single Arbiter PUF — breakable, as claimed

- 15 settings swept (5 C-values x LinearSVC, 5 C-values x LogisticRegression,
  5 degrees x polynomial SVC) → backs "across 15+ settings."
- Best result: **LinearSVC, 99.75% test accuracy**, trained in **0.33s**
  on 16,000 CRPs (80/20 split), tested on 4,000 held-out CRPs.
- This matches the resume bullet closely and is now backed by a real
  train/test split, not just training accuracy.

## 2. k=4 XOR (ML-)PUF — NOT broken by this method, and that's the real finding

- Same 64-bit challenges, response = XOR of 4 independent Arbiter PUFs.
- Attacked with a 136-dimensional expanded feature map (pairwise
  interaction terms) + degree-8 polynomial kernel SVM.
- Result: **49.65% test accuracy — statistically indistinguishable from
  random guessing (50%).**

**This is not a bug, and you should not hide it.** It matches the
published PUF-security literature: a 4-way XOR PUF is specifically
designed to resist exactly this kind of attack, and breaking it for
real typically needs either (a) millions of CRPs rather than tens of
thousands, or (b) more specialized optimization (CMA-ES-based attacks,
or deep learning with careful architecture search) rather than a
fixed-kernel SVM. A resume bullet claiming "99% accuracy on ML-PUF" via
this exact method is not defensible — don't claim it.

**What you CAN honestly say instead, which is a stronger interview
answer anyway:** *"I showed a single Arbiter PUF is fully breakable
with a simple linear model (99.75% accuracy), then attempted the same
style of attack — feature expansion + polynomial kernel — against a
4-XOR ML-PUF and found it resistant at the CRP scale I tested (near
50%, i.e. no better than guessing), which empirically demonstrates why
XOR-PUFs are the standard defense recommendation."* That's a security
project with a real negative result, which reads as more credible to
an interviewer than an implausible blanket "99% on everything."

## 3. Parameter recovery — corrected to what's actually recoverable

- For a single Arbiter PUF, a trained linear model's weight vector
  (64 stage weights + 1 bias = 65 numbers) **is already** the full set
  of "effective delay parameters" recoverable from CRPs. There is no
  deeper physical quantity to invert further — the raw two-path delay
  values are provably non-identifiable from this data (many-to-one
  collapse), so don't claim to recover them.
- The "256 parameters" claim is only meaningful at the XOR-PUF level:
  4 sub-PUFs x 64 stage weights = 256. This code recovers them
  correctly **given access to each sub-PUF's individual response**
  (only possible here because it's simulated/debug silicon — see the
  docstring in `inversion.py` for why a real black-box attack on
  physical hardware, where only the final XOR bit is visible, is a
  genuinely harder problem that needs different methods).
- Recovery was validated against known ground truth (only possible in
  simulation) with mean absolute error ≈ 1.25–1.38 per sub-PUF — report
  this as "recovered within measured error of ground truth," not as
  "exactly reconstructed."

## File map

| File | What it does |
|---|---|
| `puf_simulator.py` | Generates ground-truth PUF weights + CRPs so results can be validated |
| `feature_maps.py` | `arbiter_feature_map` (single PUF), `xor_feature_map` (expanded, 45–136 dim) |
| `models.py` | LinearSVC / LogisticRegression / polynomial-kernel SVC training |
| `inversion.py` | Honest parameter extraction + recovery, with identifiability caveats documented |
| `experiments.py` | The 15+ setting sweep with a real train/test split and timing |
| `main.py` | Runs everything end to end, prints real numbers |

## Suggested resume bullet rewrite

> Modeled a 64-bit Arbiter PUF via linear classifiers (LinearSVC,
> Logistic Regression), achieving 99.75% CRP-prediction accuracy across
> 15 hyperparameter settings; extended the attack to a 4-XOR ML-PUF
> using expanded polynomial features (up to 136-dim) and a degree-8
> kernel SVM, demonstrating the single-PUF attack's limits (near-chance
> accuracy) and recovering 256 effective stage-delay parameters (4 ×
> 64) from simulated ground truth with validated reconstruction error.
