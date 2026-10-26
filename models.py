"""
models.py
=========
Classifiers used to attack Arbiter PUFs (linear, since a single APUF
is linearly separable in feature space) and XOR/ML-PUFs (polynomial
kernel SVM, since XOR destroys that linear separability).
"""

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression


def train_linear_svc(X_train, y_train, C=1.0):
    """L2-regularized linear SVM with hinge loss. Used on a single APUF."""
    model = LinearSVC(
        C=C, loss="hinge", penalty="l2", dual=True, tol=1e-4, max_iter=20000
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, C=1.0):
    """L2-regularized Logistic Regression. Used on a single APUF."""
    model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=5000)
    model.fit(X_train, y_train)
    return model


def train_polynomial_svc(X_train, y_train, degree=8, C=1.0, gamma="scale"):
    """
    Polynomial-kernel SVM: K(x, z) = (gamma * x^T z + coef0) ** degree

    Used on the XOR-PUF, where degree=8 lets the SVM implicitly
    operate in a much higher-order interaction space than we could
    afford to materialize explicitly via xor_feature_map.
    """
    model = SVC(
        kernel="poly", degree=degree, C=C, gamma=gamma, coef0=1.0, max_iter=-1
    )
    model.fit(X_train, y_train)
    return model


def predict_apuf(model, X, feature_fn):
    """Predict responses for raw challenges using the given feature map."""
    Phi = feature_fn(X)
    return model.predict(Phi)
