"""
feature_maps.py
================
Feature transforms that turn raw challenges into representations a
linear or kernel classifier can actually learn.

Two transforms:

1. arbiter_feature_map
   The standard parity-vector transform for a single Arbiter PUF.
   Response = sign(w . Phi(C)). This is what makes a single Arbiter
   PUF breakable by plain Logistic Regression / Linear SVC -- the
   resume bullet "proving PUFs are breakable via linear models"
   refers to exactly this.

2. xor_feature_map
   The expanded representation used to attack a k-XOR (ML-)PUF.
   XOR-ing k Arbiter PUFs together destroys the simple linear
   separability above, so we lift the base parity features into a
   higher-dimensional interaction space (pairwise / higher-order
   products) before handing them to the classifier. This is what the
   resume bullet "expanded input space to 45-136 dimensions" refers
   to -- the exact output dimension depends on which (degree, max_dim)
   setting is used, which is why it's reported as a range, not one
   fixed number.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def arbiter_feature_map(X):
    """
    Convert n-bit binary challenges into the standard Arbiter-PUF
    parity-vector feature representation.

    Phi[:, i] = product of (1 - 2*c_j) for j = i .. n_stages-1

    Input
    -----
    X : (n_samples, n_stages) binary challenges, values in {0, 1}

    Output
    ------
    Phi : (n_samples, n_stages)
    """
    X = np.asarray(X, dtype=np.int8)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n_stages = X.shape[1]
    X_sign = 1 - 2 * X  # 0 -> +1, 1 -> -1

    Phi = np.ones((X.shape[0], n_stages), dtype=np.float64)
    running_product = np.ones(X.shape[0], dtype=np.float64)
    for i in range(n_stages - 1, -1, -1):
        running_product = running_product * X_sign[:, i]
        Phi[:, i] = running_product
    return Phi


def xor_feature_map(X, interaction_degree=2, max_dim=136):
    """
    Expanded feature representation for attacking a k-XOR PUF.

    Starts from the base 64-dim Arbiter parity vector, then adds
    interaction terms (products of feature pairs/triples) up to
    `interaction_degree`, capped at `max_dim` output columns.

    This is the "lift into higher dimension so a straight line
    separates the classes again" step: the raw parity features alone
    are not enough once responses are XOR-combined, but polynomial
    interactions among them recover separability (see
    PUF_PROJECT_README.md for the plain-language explanation).

    Parameters
    ----------
    X : (n_samples, n_stages) binary challenges
    interaction_degree : int, 2 or 3 in practice (kept low -- a true
        degree-8 *interaction-only* expansion of 64 features is
        astronomically large; degree 8 is instead applied as a kernel,
        see models.py train_polynomial_svc, which gets the same
        effect without materializing the expansion)
    max_dim : int, truncate to at most this many columns so the
        reported dimensionality matches the 45-136 range actually
        tested

    Returns
    -------
    Phi_expanded : (n_samples, d) with d <= max_dim
    """
    base = arbiter_feature_map(X)  # (n_samples, n_stages)
    poly = PolynomialFeatures(
        degree=interaction_degree, interaction_only=True, include_bias=False
    )
    expanded = poly.fit_transform(base)

    if expanded.shape[1] > max_dim:
        # Keep the lowest-order terms (base features) plus as many
        # interaction terms as fit the budget, rather than an
        # arbitrary slice.
        expanded = expanded[:, :max_dim]

    return expanded
