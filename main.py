"""
main.py
=======
Runs the full pipeline end to end and prints the numbers you should
actually quote on your resume/in interviews -- replace any resume
bullet number with what this prints on your machine, not the original
unverified figures.
"""

import numpy as np

from puf_simulator import (
    generate_random_weights, generate_challenges,
    simulate_arbiter_puf, simulate_xor_puf,
)
from experiments import run_apuf_experiments, run_xor_puf_experiment
from models import train_linear_svc
from inversion import extract_linear_model_parameters, recover_xor_stage_weights, recovery_error
from feature_maps import arbiter_feature_map


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


if __name__ == "__main__":
    N_STAGES = 64
    N_SAMPLES = 20000
    K_SUBPUFS = 4
    SEED = 42

    # ----------------------------------------------------------------
    section("1. Single Arbiter PUF: modeling attack")
    # ----------------------------------------------------------------
    true_w = generate_random_weights(N_STAGES, seed=SEED)
    X = generate_challenges(N_SAMPLES, N_STAGES, seed=SEED)
    y = simulate_arbiter_puf(X, true_w)

    apuf_results = run_apuf_experiments(X, y, seed=SEED)
    best = max(apuf_results, key=lambda r: r["accuracy"])
    print(f"Ran {len(apuf_results)} settings "
          f"(backs the '{len(apuf_results)}+ settings' claim).")
    print(f"Best: {best['model']} C={best['C']} degree={best['degree']} "
          f"-> {best['accuracy']*100:.2f}% test accuracy, "
          f"trained in {best['train_time_s']:.3f}s "
          f"on {best['n_train']} CRPs, tested on {best['n_test']}.")

    # ----------------------------------------------------------------
    section("2. k-XOR (ML-)PUF: modeling attack with expanded features")
    # ----------------------------------------------------------------
    sub_weights_true = [generate_random_weights(N_STAGES, seed=SEED + i)
                         for i in range(K_SUBPUFS)]
    X_xor = generate_challenges(N_SAMPLES, N_STAGES, seed=SEED + 100)
    y_xor, sub_responses = simulate_xor_puf(X_xor, sub_weights_true)

    xor_result = run_xor_puf_experiment(X_xor, y_xor, interaction_degree=2,
                                         max_dim=136, kernel_degree=8, seed=SEED)
    print(f"XOR-PUF (k={K_SUBPUFS}) attacked via {xor_result['feature_dim']}-dim "
          f"expanded features + degree-8 poly kernel:")
    print(f"  -> {xor_result['accuracy']*100:.2f}% test accuracy, "
          f"trained in {xor_result['train_time_s']:.3f}s "
          f"on {xor_result['n_train']} CRPs.")

    # ----------------------------------------------------------------
    section("3. Parameter recovery (honest version)")
    # ----------------------------------------------------------------
    # Train one linear model PER sub-PUF using its own response label
    # (available here because this is simulated/debug silicon -- see
    # inversion.py docstring for why this does NOT carry over directly
    # to a real black-box attack on physical hardware).
    sub_models = []
    for i in range(K_SUBPUFS):
        Phi = arbiter_feature_map(X_xor)
        model = train_linear_svc(Phi, sub_responses[:, i])
        sub_models.append(model)

    stage_weights, biases = recover_xor_stage_weights(sub_models)
    print(f"Recovered stage-weight matrix: {stage_weights.shape} "
          f"= {stage_weights.size} total effective delay parameters "
          f"({K_SUBPUFS} sub-PUFs x {N_STAGES} stages).")

    errors = recovery_error(stage_weights, sub_weights_true)
    for i, err in enumerate(errors):
        print(f"  Sub-PUF {i+1}: mean abs error vs. ground truth = {err:.4f}")
    print("(Low error here is only checkable because we simulated the "
          "ground truth -- that's the honest validation this project "
          "needed and the original code never did.)")
