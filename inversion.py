"""
inversion.py
============
Recovering the PUF's internal parameters from a trained model.

IMPORTANT -- read before quoting numbers on a resume
-------------------------------------------------------
A linear model trained on the Arbiter parity features learns exactly
65 numbers: 64 stage weights + 1 bias. Those 65 numbers ARE the full
set of "effective delay parameters" that any CRP-based attack can
ever recover for a single Arbiter PUF -- the underlying 128 raw
physical delays (2 per stage) collapse into those 65 under the linear
delay model, and that collapse is many-to-one. No amount of extra
algebra on top of the 65 learned weights recovers the missing
information; it was never observable from challenge-response data in
the first place. (This is a known identifiability result in the PUF
literature, not a limitation of this code.)

So: "256 recovered hardware delay parameters" is only a meaningful,
defensible claim at the XOR-PUF level, where you train k=4 SEPARATE
Arbiter models (one per sub-PUF) and stack their 4 x 64 = 256 stage
weights together. That is what recover_xor_stage_weights below
actually does -- and it requires access to each sub-PUF's individual
response, not just the final XOR bit.
"""

import numpy as np


def extract_linear_model_parameters(model):
    """
    Pull the learned weight vector + bias out of a trained linear
    classifier (LinearSVC or LogisticRegression).

    Returns
    -------
    params : (n_features + 1,) = [w_1, ..., w_n, bias]
    """
    if not hasattr(model, "coef_"):
        raise ValueError("Model must be a linear classifier with coef_.")

    weights = model.coef_.reshape(-1)
    bias = float(model.intercept_[0])
    return np.concatenate([weights, [bias]])


def recover_xor_stage_weights(sub_models):
    """
    Recover the k x n_stages = "256" effective stage-weight parameters
    for a k-XOR PUF, GIVEN k separately trained single-APUF models
    (one per sub-PUF).

    Honesty note: this assumes you can train each sub-PUF model
    independently, which requires per-sub-PUF response labels. In a
    simulated/debug setting (puf_simulator.simulate_xor_puf returns
    sub_responses for exactly this reason) that's straightforward. In
    a real black-box attack against silicon where only the final XOR
    bit is observable, separating the k sub-PUFs is a genuinely harder
    problem solved in the literature with evolutionary strategies
    (CMA-ES over the whole XOR-PUF, not independent linear fits) --
    that is out of scope here and should not be claimed as done.

    Parameters
    ----------
    sub_models : list of k trained linear models (one per sub-PUF)

    Returns
    -------
    stage_weights : (k, n_stages) -- the 4 x 64 = 256 recovered values
    biases : (k,) -- the k bias terms, reported separately
    """
    all_weights = []
    all_biases = []
    for m in sub_models:
        params = extract_linear_model_parameters(m)
        all_weights.append(params[:-1])
        all_biases.append(params[-1])

    stage_weights = np.vstack(all_weights)  # (k, n_stages)
    biases = np.array(all_biases)
    return stage_weights, biases


def recovery_error(stage_weights_recovered, ground_truth_weights_list):
    """
    Validate the recovery against known ground truth (only possible in
    simulation, which is exactly why puf_simulator.py exists -- you
    cannot validate an inversion attack without knowing the answer).

    Returns mean absolute error per sub-PUF, after resolving the sign
    ambiguity (a PUF and its "mirror" with every weight negated
    produce identical responses, since sign(w.Phi) = sign(-w.-Phi)
    only up to relabeling -- so we align signs before comparing).
    """
    errors = []
    for w_hat, w_true in zip(stage_weights_recovered, ground_truth_weights_list):
        w_true_stages = np.asarray(w_true)[:-1]  # drop bias for this comparison
        # resolve sign ambiguity
        if np.dot(w_hat, w_true_stages) < 0:
            w_hat = -w_hat
        errors.append(np.mean(np.abs(w_hat - w_true_stages)))
    return np.array(errors)
