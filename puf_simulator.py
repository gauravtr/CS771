"""
puf_simulator.py
=================
Ground-truth simulator for Arbiter PUFs and k-XOR (ML-)PUFs.

Why this file exists
---------------------
To honestly claim "99% accuracy" or "trained on N challenge-response
pairs" you need a labelled dataset. Real silicon isn't available here,
so we simulate ground-truth PUF behaviour the same way the PUF
modelling-attack literature does (Ruehrmair et al.): sample a random
"effective delay weight" vector per Arbiter PUF and generate exact
responses from it. This also gives us ground truth to validate the
parameter-recovery step against (Section 4), which is the only
rigorous way to check an inversion attack actually works.

Note on what is and isn't physically realistic
------------------------------------------------
A real Arbiter PUF's response comes from 2 raw delay values per stage
(2*64 = 128 numbers). Those collapse into a 65-dimensional "effective
weight" vector w (64 stage terms + 1 bias) under the standard linear
delay model -- this collapse is a many-to-one mapping, not a bijection.
That means: even with perfect knowledge of w, the original 128 raw
delays are NOT uniquely recoverable. This is a known identifiability
limit of the linear APUF model, not a bug. We simulate directly at the
w-level because that is the maximal level of detail any CRP-based
attack (including the original resume claim) can ever recover.
"""

import numpy as np
from feature_maps import arbiter_feature_map


def generate_random_weights(n_stages=64, scale=1.0, seed=None):
    """
    Sample a random ground-truth effective weight vector for one
    Arbiter PUF: n_stages stage weights + 1 bias term.

    Returns
    -------
    w : shape (n_stages + 1,)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=scale, size=n_stages + 1)


def generate_challenges(n_samples, n_stages=64, seed=None):
    """Random binary challenges, shape (n_samples, n_stages)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_samples, n_stages)).astype(np.int8)


def simulate_arbiter_puf(X, weights):
    """
    Exact (noise-free) ground-truth response of a single Arbiter PUF.

    response = sign( [Phi(X), 1] . weights )

    Parameters
    ----------
    X : (n_samples, n_stages) binary challenges
    weights : (n_stages + 1,) ground-truth effective weights

    Returns
    -------
    y : (n_samples,) in {0, 1}
    """
    Phi = arbiter_feature_map(X)
    Phi_aug = np.hstack([Phi, np.ones((Phi.shape[0], 1))])
    delay = Phi_aug @ weights
    return (delay < 0).astype(np.int8)  # sign(delay) -> 0/1


def simulate_xor_puf(X, weights_list):
    """
    Exact ground-truth response of a k-XOR (ML-)PUF: XOR of k
    independent Arbiter PUF responses to the same challenge.

    Parameters
    ----------
    X : (n_samples, n_stages) binary challenges
    weights_list : list of k ground-truth weight vectors,
                   one per sub-PUF, each (n_stages + 1,)

    Returns
    -------
    y : (n_samples,) in {0, 1}
    sub_responses : (n_samples, k) individual sub-PUF bits
                     (only observable on simulated/debug silicon,
                     NOT in a real black-box attack -- see inversion.py)
    """
    sub_responses = np.column_stack(
        [simulate_arbiter_puf(X, w) for w in weights_list]
    )
    y = np.bitwise_xor.reduce(sub_responses, axis=1)
    return y, sub_responses
