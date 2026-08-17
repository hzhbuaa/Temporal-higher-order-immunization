"""Mean-field stationary solution of SIS dynamics on a HOAD temporal network."""

import argparse
import json

import numpy as np
from scipy.optimize import root

from utils import (
    STRATEGIES,
    activity_grid,
    check_strategy,
    immunization_probabilities,
    resolve_set_fraction,
)


def find_equilibria(
    a1,
    a2,
    mass,
    strategy="R",
    immune_fraction=0.1,
    lambda1=3.4,
    lambda2=60.0,
    rho0=0.2,
    initial_guesses=12,
    mu=0.01,
    root_method="hybr",
):
    """Find Eq. (21) equilibria using the ``theory_base.py`` root workflow.

    ``mapped_rho0`` is the paper's pre-immunization counterpart
    integral i_plus(a)/(1-q(a)) da from Eqs. (22)-(23).
    """
    strategy = check_strategy(strategy)
    a1 = np.asarray(a1, dtype=float).ravel()
    a2 = np.asarray(a2, dtype=float).ravel()
    mass = np.asarray(mass, dtype=float).ravel()
    if not (a1.shape == a2.shape == mass.shape):
        raise ValueError("a1, a2 and mass must have identical shapes")
    mass = mass / mass.sum()
    set_fraction = resolve_set_fraction(
        strategy, a1, a2, mass, immune_fraction, lambda1, lambda2, rho0
    )
    q = immunization_probabilities(
        strategy, a1, a2, mass, set_fraction, lambda1, lambda2, rho0
    )
    remaining = mass * (1.0 - q)

    beta1 = float(lambda1) * float(mu)
    beta2 = float(lambda2) * float(mu)

    def profile_and_residual(x):
        rho, theta1, theta2 = map(float, x)
        if rho < 0.0:
            rho = 0.0
        infection_intensity = (
            beta1 * (a1 * rho + theta1)
            + beta2 * (a2 * rho * rho + 2.0 * theta2 * rho)
        )
        infection_intensity = np.maximum(infection_intensity, 0.0)
        profile = remaining * infection_intensity / (float(mu) + infection_intensity)
        residual = np.array(
            [
                profile.sum() - rho,
                np.dot(a1, profile) - theta1,
                np.dot(a2, profile) - theta2,
            ],
            dtype=float,
        )
        return profile, residual

    def profile_and_residual_as_vector(x):
        return profile_and_residual(x)[1]

    roots = [
        {
            "moments": np.zeros(3),
            "post_profile": np.zeros_like(mass),
            "pre_profile": np.zeros_like(mass),
            "mapped_rho0": 0.0,
            "residual_norm": 0.0,
            "converged": True,
            "iterations": 0,
            "initial_fraction": 0.0,
            "rounded_prevalence": 0.0,
            "rounded_mapped_rho0": 0.0,
        }
    ]
    rounded_solutions = {0.0}
    rounded_mapped_solutions = {0.0}
    fractions = np.linspace(0.001, 1.0, max(1, int(initial_guesses)))
    for fraction in fractions:
        initial_profile = remaining * fraction
        seed = np.array(
            [
                initial_profile.sum(),
                np.dot(a1, initial_profile),
                np.dot(a2, initial_profile),
            ],
            dtype=float,
        )
        solved = root(profile_and_residual_as_vector, seed, method=root_method)
        if not solved.success:
            continue
        moments = np.asarray(solved.x, dtype=float)
        if moments[0] < 0.0:
            moments[0] = 0.0
        post_profile, residual = profile_and_residual(moments)
        remaining_check = np.maximum(remaining, 0.0)
        if (
            not np.all(np.isfinite(post_profile))
            or np.any(post_profile < -1e-12)
            or np.any(post_profile - remaining_check > 1e-10)
        ):
            continue
        rounded_prevalence = round(float(post_profile.sum()), 7)
        denominator = 1.0 - q
        pre_profile = np.divide(
            post_profile,
            denominator,
            out=np.zeros_like(post_profile),
            where=denominator > 0.0,
        )
        mapped_rho0 = float(pre_profile.sum())
        rounded_mapped_rho0 = round(mapped_rho0, 7)
        rounded_mapped_solutions.add(rounded_mapped_rho0)
        if rounded_prevalence in rounded_solutions:
            if len(rounded_solutions) >= 3 and len(rounded_mapped_solutions) >= 3:
                break
            continue
        rounded_solutions.add(rounded_prevalence)
        roots.append(
            {
                "moments": moments,
                "post_profile": post_profile,
                "pre_profile": pre_profile,
                "mapped_rho0": mapped_rho0,
                "residual_norm": float(np.linalg.norm(residual, ord=np.inf)),
                "converged": True,
                "iterations": int(getattr(solved, "nfev", 0)),
                "initial_fraction": float(fraction),
                "rounded_prevalence": rounded_prevalence,
                "rounded_mapped_rho0": rounded_mapped_rho0,
            }
        )
        if len(rounded_solutions) >= 3 and len(rounded_mapped_solutions) >= 3:
            break

    roots.sort(key=lambda item: float(item["moments"][0]))
    for index, item in enumerate(roots):
        prevalence = float(item["moments"][0])
        if round(prevalence, 7) == 0.0:
            branch = "disease_free"
        elif index == len(roots) - 1:
            branch = "endemic"
        else:
            branch = "intermediate"
        item.update(
            {
                "branch": branch,
                "prevalence": prevalence,
                "theta1": float(item["moments"][1]),
                "theta2": float(item["moments"][2]),
            }
        )
    return {
        "strategy": strategy,
        "set_fraction": float(set_fraction),
        "immune_fraction": float(immune_fraction),
        "equilibria": roots,
    }


def equilibrium_summary(result):
    """Return a JSON-serializable summary without the activity-class profiles."""
    return {
        "strategy": result["strategy"],
        "set_fraction": result["set_fraction"],
        "immune_fraction": result["immune_fraction"],
        "equilibria": [
            {
                key: value
                for key, value in equilibrium.items()
                if key not in ("moments", "post_profile", "pre_profile")
            }
            for equilibrium in result["equilibria"]
        ],
    }


def stationary_prevalence(
    a1,
    a2,
    mass,
    strategy="R",
    immune_fraction=0.1,
    lambda1=3.4,
    lambda2=60.0,
    rho0=0.2,
    initial_prevalence=0.2,
    tolerance=1e-12,
    max_iterations=20000,
):
    """Solve the three-moment HOAD fixed-point equations by damped iteration."""
    strategy = check_strategy(strategy)
    a1 = np.asarray(a1, dtype=float).ravel()
    a2 = np.asarray(a2, dtype=float).ravel()
    mass = np.asarray(mass, dtype=float).ravel()
    mass = mass / mass.sum()
    set_fraction = resolve_set_fraction(
        strategy, a1, a2, mass, immune_fraction, lambda1, lambda2, rho0
    )
    q = immunization_probabilities(
        strategy, a1, a2, mass, set_fraction, lambda1, lambda2, rho0
    )
    remaining = mass * (1.0 - q)

    infected = remaining * float(initial_prevalence)
    damping = 0.35
    for iteration in range(1, int(max_iterations) + 1):
        rho = float(infected.sum())
        theta1 = float(np.dot(a1, infected))
        theta2 = float(np.dot(a2, infected))
        force = (
            float(lambda1) * (a1 * rho + theta1)
            + float(lambda2) * (a2 * rho * rho + 2.0 * theta2 * rho)
        )
        updated = remaining * force / (1.0 + force)
        next_infected = (1.0 - damping) * infected + damping * updated
        if np.max(np.abs(next_infected - infected)) < tolerance:
            infected = next_infected
            break
        infected = next_infected
    else:
        raise RuntimeError("the stationary iteration did not converge")

    return {
        "strategy": strategy,
        "prevalence": float(infected.sum()),
        "set_fraction": float(set_fraction),
        "immune_fraction": float(immune_fraction),
        "iterations": iteration,
    }


def run_theory(
    strategy="R",
    eta1=2.1,
    mean1=0.13,
    eta2=2.1,
    mean2=0.03,
    bins=120,
    **kwargs
):
    a1, a2, mass = activity_grid(eta1, mean1, eta2, mean2, bins)
    return stationary_prevalence(a1, a2, mass, strategy=strategy, **kwargs)


def run_equilibria(
    strategy="R",
    eta1=2.1,
    mean1=0.13,
    eta2=2.1,
    mean2=0.03,
    bins=120,
    **kwargs
):
    a1, a2, mass = activity_grid(eta1, mean1, eta2, mean2, bins)
    return find_equilibria(a1, a2, mass, strategy=strategy, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="R", choices=list(STRATEGIES) + ["all"])
    parser.add_argument("--immune-fraction", type=float, default=0.1)
    parser.add_argument("--lambda1", type=float, default=3.4)
    parser.add_argument("--lambda2", type=float, default=60.0)
    parser.add_argument("--rho0", type=float, default=0.2)
    parser.add_argument("--initial-prevalence", type=float, default=0.2)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--all-equilibria", action="store_true")
    parser.add_argument("--initial-guesses", type=int, default=12)
    args = parser.parse_args()
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    for strategy in strategies:
        common = dict(
            strategy=strategy, immune_fraction=args.immune_fraction,
            lambda1=args.lambda1, lambda2=args.lambda2,
            rho0=args.rho0, bins=args.bins
        )
        if args.all_equilibria:
            result = equilibrium_summary(
                run_equilibria(initial_guesses=args.initial_guesses, **common)
            )
        else:
            result = run_theory(
                initial_prevalence=args.initial_prevalence, **common
            )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
