# PUF Modeling and Inversion

Course Project | CS771 | Prof. Purushottam Kar | IIT Kanpur (Aug'25 - Nov'25)

## Objective

Model and invert a 64-bit Arbiter PUF and ML-PUF, demonstrating that enhanced PUFs can be broken using linear models.

## Approach

- Expanded the input feature space from 64 to 45-136 dimensions using pairwise, triple, and quadruple interaction terms of sign-converted challenge bits.
- Proposed a degree-8 polynomial kernel formulation to accurately capture PUF monomials.
- Reconstructed 256 delay parameters via a 64+1 linear model, testing SVC and Logistic Regression across 15+ hyperparameter settings.

## Results

- Achieved 99% accuracy using L1-regularized Linear SVC (hinge loss, L2 penalty) and Logistic Regression (L2 penalty).
- Training time of just 0.3 seconds.
