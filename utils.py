"""Shared activity-distribution and immunization utilities for the HOAD model."""

import numpy as np


STRATEGIES = ("R", "HIC", "TA", "HA", "PA", "EHS", "EPS", "EBS")


def check_strategy(strategy):
    strategy = str(strategy).upper()
    if strategy not in STRATEGIES:
        raise ValueError("strategy must be one of: " + ", ".join(STRATEGIES))
    return strategy


def _power_law_mean(epsilon, eta):
    normalizer = (1.0 - eta) / (1.0 - epsilon ** (1.0 - eta))
    if abs(eta - 2.0) < 1e-12:
        return -normalizer * np.log(epsilon)
    return normalizer * (1.0 - epsilon ** (2.0 - eta)) / (2.0 - eta)


def solve_lower_cutoff(eta, target_mean, tol=1e-13):
    """Find the lower cutoff of a power law p(a) proportional to a**(-eta)."""
    eta = float(eta)
    target_mean = float(target_mean)
    if eta <= 1.0 or not 0.0 < target_mean < 1.0:
        raise ValueError("eta must exceed 1 and target_mean must lie in (0, 1)")

    lo, hi = 1e-12, 1.0 - 1e-12
    if not _power_law_mean(lo, eta) <= target_mean <= _power_law_mean(hi, eta):
        raise ValueError("the requested mean is outside the supported range")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _power_law_mean(mid, eta) < target_mean:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def power_law_quantiles(eta, target_mean, size, rng=None, deterministic=False):
    """Return samples (or equal-probability midpoints) from the activity law."""
    size = int(size)
    if size <= 0:
        raise ValueError("size must be positive")
    epsilon = solve_lower_cutoff(eta, target_mean)
    if deterministic:
        u = (np.arange(size, dtype=float) + 0.5) / size
    else:
        rng = np.random.default_rng() if rng is None else rng
        u = rng.random(size)
    base = epsilon ** (1.0 - eta) + u * (1.0 - epsilon ** (1.0 - eta))
    return base ** (1.0 / (1.0 - eta))


def activity_grid(eta1, mean1, eta2, mean2, bins):
    """Construct an independent, equal-probability (a1, a2) quadrature grid."""
    a1 = power_law_quantiles(eta1, mean1, bins, deterministic=True)
    a2 = power_law_quantiles(eta2, mean2, bins, deterministic=True)
    a1_grid, a2_grid = np.meshgrid(a1, a2, indexing="ij")
    mass = np.full(a1_grid.size, 1.0 / a1_grid.size)
    return a1_grid.ravel(), a2_grid.ravel(), mass


def _allocate_top_scores(score, mass, fraction):
    q = np.zeros_like(mass, dtype=float)
    remaining = float(np.clip(fraction, 0.0, 1.0))
    for idx in np.argsort(score)[::-1]:
        if remaining <= 0.0:
            break
        take = min(float(mass[idx]), remaining)
        if mass[idx] > 0.0:
            q[idx] = take / mass[idx]
        remaining -= take
    return np.where(q > 1.0 - 1e-13, 1.0 - 1e-8, q)


def immunization_probabilities(
    strategy, a1, a2, mass, fraction, lambda1, lambda2, rho0
):
    """Return q(a1,a2), the immunization probability for each activity type.

    Naming used in the accompanying paper code:
      R: random; TA: descending a1+a2; HA: descending a2;
      PA: descending a1; HIC: descending infection kernel;
      EPS/EHS/EBS: pairwise/higher-order/both egocentric sampling.
    """
    strategy = check_strategy(strategy)
    a1 = np.asarray(a1, dtype=float).ravel()
    a2 = np.asarray(a2, dtype=float).ravel()
    mass = np.asarray(mass, dtype=float).ravel()
    if not (a1.shape == a2.shape == mass.shape):
        raise ValueError("a1, a2 and mass must have identical shapes")
    total = mass.sum()
    if total <= 0.0:
        raise ValueError("mass must have a positive sum")
    mass = mass / total
    phi = float(np.clip(fraction, 0.0, 1.0))

    if strategy == "R":
        return np.full_like(mass, phi)

    if strategy in ("TA", "HA", "PA", "HIC"):
        if strategy == "TA":
            score = a1 + a2
        elif strategy == "HA":
            score = a2
        elif strategy == "PA":
            score = a1
        else:
            rho_star = float(rho0) * (1.0 - phi)
            score = 2.0 * float(lambda1) * a1 + 3.0 * float(lambda2) * rho_star * a2
        return _allocate_top_scores(score, mass, phi)

    mean1 = float(np.dot(mass, a1))
    mean2 = float(np.dot(mass, a2))
    if strategy == "EPS":
        inv = np.divide(1.0, a1 + mean1, out=np.zeros_like(a1), where=a1 + mean1 > 0)
        exposure = np.dot(mass, a1 * inv) + a1 * np.dot(mass, inv)
    elif strategy == "EHS":
        inv = np.divide(1.0, a2 + 2.0 * mean2, out=np.zeros_like(a2), where=a2 + 2.0 * mean2 > 0)
        exposure = np.dot(mass, a2 * inv) + (a2 + mean2) * np.dot(mass, inv)
    else:  # EBS
        source = a1 + 2.0 * a2
        inv = np.divide(
            1.0,
            source + mean1 + 4.0 * mean2,
            out=np.zeros_like(source),
            where=source + mean1 + 4.0 * mean2 > 0,
        )
        exposure = np.dot(mass, source * inv) + (source + 2.0 * mean2) * np.dot(mass, inv)
    return np.clip(1.0 - np.exp(-phi * exposure), 0.0, 1.0)


def resolve_set_fraction(
    strategy,
    a1,
    a2,
    mass,
    target_immune_fraction,
    lambda1,
    lambda2,
    rho0,
    tolerance=5e-15,
):
    """Convert an actual immune fraction omega to the strategy input fraction.

    For R/HIC/TA/HA/PA the two fractions are identical. For EPS/EHS/EBS,
    bisection finds the probe fraction phi satisfying sum(q(phi) * mass) = omega,
    matching the parameter convention used by ``theory/theory_base.py``.
    """
    strategy = check_strategy(strategy)
    target = float(target_immune_fraction)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target_immune_fraction must lie in [0, 1]")
    if strategy not in ("EPS", "EHS", "EBS") or target == 0.0:
        return target

    mass = np.asarray(mass, dtype=float).ravel()
    mass = mass / mass.sum()

    def actual(probe_fraction):
        q = immunization_probabilities(
            strategy, a1, a2, mass, probe_fraction, lambda1, lambda2, rho0
        )
        return float(np.dot(mass, q))

    lo, hi = target, 1.0
    if actual(hi) + tolerance < target:
        raise ValueError(
            f"target immune fraction {target} is unreachable for {strategy} "
            "with a probe fraction no larger than one"
        )
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        value = actual(mid)
        if round(value, 15) == round(target, 15):
            return mid
        if value < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
