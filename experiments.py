"""
experiments.py
===============
Runs the sweep across model types / hyperparameters referenced by the
"across 15+ settings" resume bullet, on a proper held-out test split,
and times training (backs the "0.3s training time" bullet -- actual
number depends on your machine, print it and use your real value).
"""

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

from feature_maps import arbiter_feature_map, xor_feature_map


def run_apuf_experiments(X, y, test_size=0.2, seed=0):
    """
    Sweep of LinearSVC / LogisticRegression / polynomial-kernel SVC
    over multiple C values and kernel degrees, on a single Arbiter
    PUF's CRPs, with a held-out test set.

    Returns a list of dicts: one row per (model, setting, accuracy,
    train_time_seconds, n_train, n_test) -- everything you need to
    honestly defend "99% accuracy across 15+ settings" with numbers
    you can actually quote.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    Phi_train = arbiter_feature_map(X_train)
    Phi_test = arbiter_feature_map(X_test)

    results = []

    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        t0 = time.perf_counter()
        model = LinearSVC(C=C, loss="hinge", penalty="l2", dual=True,
                           tol=1e-4, max_iter=20000)
        model.fit(Phi_train, y_train)
        train_time = time.perf_counter() - t0
        acc = accuracy_score(y_test, model.predict(Phi_test))
        results.append(dict(model="LinearSVC", C=C, degree=None, accuracy=acc,
                             train_time_s=train_time,
                             n_train=len(y_train), n_test=len(y_test)))

    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        t0 = time.perf_counter()
        model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=5000)
        model.fit(Phi_train, y_train)
        train_time = time.perf_counter() - t0
        acc = accuracy_score(y_test, model.predict(Phi_test))
        results.append(dict(model="LogisticRegression", C=C, degree=None, accuracy=acc,
                             train_time_s=train_time,
                             n_train=len(y_train), n_test=len(y_test)))

    for degree in [2, 4, 6, 8, 10]:
        t0 = time.perf_counter()
        model = SVC(kernel="poly", degree=degree, C=1.0, gamma="scale", coef0=1.0)
        model.fit(Phi_train, y_train)
        train_time = time.perf_counter() - t0
        acc = accuracy_score(y_test, model.predict(Phi_test))
        results.append(dict(model="PolynomialSVC", C=1.0, degree=degree, accuracy=acc,
                             train_time_s=train_time,
                             n_train=len(y_train), n_test=len(y_test)))

    return results


def run_xor_puf_experiment(X, y, interaction_degree=2, max_dim=136,
                            kernel_degree=8, test_size=0.2, seed=0):
    """
    Single run attacking a k-XOR PUF's final (already-XORed) response
    using the expanded feature map + polynomial kernel SVC.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    Phi_train = xor_feature_map(X_train, interaction_degree, max_dim)
    Phi_test = xor_feature_map(X_test, interaction_degree, max_dim)

    t0 = time.perf_counter()
    model = SVC(kernel="poly", degree=kernel_degree, C=1.0, gamma="scale", coef0=1.0)
    model.fit(Phi_train, y_train)
    train_time = time.perf_counter() - t0

    acc = accuracy_score(y_test, model.predict(Phi_test))
    return dict(
        feature_dim=Phi_train.shape[1], accuracy=acc, train_time_s=train_time,
        n_train=len(y_train), n_test=len(y_test),
    )
