"""Stochastic SIS simulation on an annealed HOAD temporal network."""

import argparse
import json

import numpy as np

from utils import (
    STRATEGIES,
    check_strategy,
    immunization_probabilities,
    power_law_quantiles,
    resolve_set_fraction,
)


def _partners_excluding_source(sources, size, rng):
    raw = rng.integers(0, size - 1, size=len(sources))
    return raw + (raw >= sources)


def _second_partner(sources, first, size, rng):
    low = np.minimum(sources, first)
    high = np.maximum(sources, first)
    raw = rng.integers(0, size - 2, size=len(sources))
    return raw + (raw >= low) + (raw >= high - 1)


def _egocentric_immune(a1, a2, fraction, strategy, observation_steps, rng):
    """Select neighbors of sampled ego nodes from observed HOAD interactions."""
    size = a1.size
    ego_count = min(int(size * fraction), size)
    immune = np.zeros(size, dtype=bool)
    if ego_count <= 0 or observation_steps <= 0:
        return immune
    egos = rng.choice(size, size=ego_count, replace=False)
    ego_position = np.full(size, -1, dtype=np.int32)
    ego_position[egos] = np.arange(ego_count, dtype=np.int32)
    weights = [dict() for _ in range(ego_count)]

    def add(focal, neighbor):
        position = int(ego_position[focal])
        if position >= 0 and focal != neighbor:
            weights[position][int(neighbor)] = weights[position].get(int(neighbor), 0) + 1

    for _ in range(int(observation_steps)):
        if strategy in ("EPS", "EBS"):
            source = np.flatnonzero(rng.random(size) < a1)
            partner = _partners_excluding_source(source, size, rng)
            for u, v in zip(source, partner):
                add(u, v)
                add(v, u)
        if strategy in ("EHS", "EBS"):
            source = np.flatnonzero(rng.random(size) < a2)
            first = _partners_excluding_source(source, size, rng)
            second = _second_partner(source, first, size, rng)
            for u, v, w in zip(source, first, second):
                add(u, v)
                add(u, w)
                add(v, u)
                add(v, w)
                add(w, u)
                add(w, v)

    for candidates in weights:
        if not candidates:
            continue
        nodes = np.fromiter(candidates.keys(), dtype=np.int32)
        probability = np.fromiter(candidates.values(), dtype=float)
        probability /= probability.sum()
        immune[rng.choice(nodes, p=probability)] = True
    return immune


def _choose_immune(a1, a2, q, fraction, strategy, observation_steps, rng):
    if strategy in ("EPS", "EHS", "EBS"):
        return _egocentric_immune(
            a1, a2, fraction, strategy, observation_steps, rng
        )
    if strategy == "R":
        immune = np.zeros(q.size, dtype=bool)
        count = min(int(q.size * fraction), q.size)
        immune[rng.choice(q.size, size=count, replace=False)] = True
        return immune
    return rng.random(q.size) < q


def _single_run(a1, a2, q, immune_fraction, strategy, observation_steps,
                beta1, beta2, mu, initial_prevalence, steps, burn_in, rng):
    size = a1.size
    state = np.zeros(size, dtype=np.uint8)  # 0=S, 1=I, 2=immune
    state[_choose_immune(
        a1, a2, q, immune_fraction, strategy, observation_steps, rng
    )] = 2
    susceptible = np.flatnonzero(state == 0)
    state[susceptible[rng.random(susceptible.size) < initial_prevalence]] = 1
    samples = []

    edge_exposure = np.zeros(size, dtype=np.int32)
    triangle_exposure = np.zeros(size, dtype=np.int32)
    for t in range(int(steps)):
        infected = state == 1
        susceptible_mask = state == 0
        if not infected.any():
            samples.extend([0.0] * max(0, steps - max(t, burn_in)))
            break

        edge_exposure.fill(0)
        triangle_exposure.fill(0)

        sources = np.flatnonzero(rng.random(size) < a1)
        if sources.size:
            partners = _partners_excluding_source(sources, size, rng)
            mask = infected[sources] & susceptible_mask[partners]
            np.add.at(edge_exposure, partners[mask], 1)
            mask = infected[partners] & susceptible_mask[sources]
            np.add.at(edge_exposure, sources[mask], 1)

        sources = np.flatnonzero(rng.random(size) < a2)
        if sources.size:
            first = _partners_excluding_source(sources, size, rng)
            second = _second_partner(sources, first, size, rng)
            vertices = (sources, first, second)
            for focal, other1, other2 in (
                (vertices[0], vertices[1], vertices[2]),
                (vertices[1], vertices[0], vertices[2]),
                (vertices[2], vertices[0], vertices[1]),
            ):
                mask = susceptible_mask[focal] & infected[other1] & infected[other2]
                np.add.at(triangle_exposure, focal[mask], 1)

        at_risk = np.flatnonzero(
            susceptible_mask & ((edge_exposure + triangle_exposure) > 0)
        )
        newly_infected = np.zeros(size, dtype=bool)
        if at_risk.size:
            probability = 1.0 - (
                (1.0 - beta1) ** edge_exposure[at_risk]
                * (1.0 - beta2) ** triangle_exposure[at_risk]
            )
            selected = at_risk[rng.random(at_risk.size) < probability]
            state[selected] = 1
            newly_infected[selected] = True

        recovering = np.flatnonzero(infected & ~newly_infected)
        state[recovering[rng.random(recovering.size) < mu]] = 0
        if t >= burn_in:
            samples.append(float(np.count_nonzero(state == 1)) / size)

    return float(np.mean(samples)) if samples else 0.0, float(np.mean(state == 2))


def run_simulation(
    strategy="R",
    size=1200,
    steps=3000,
    burn_in=2000,
    repeats=3,
    observation_steps=300,
    eta1=2.1,
    mean1=0.13,
    eta2=2.1,
    mean2=0.03,
    immune_fraction=0.1,
    lambda1=3.4,
    lambda2=60.0,
    mu=0.01,
    rho0=0.2,
    initial_prevalence=0.2,
    seed=2026,
):
    strategy = check_strategy(strategy)
    if not 0 < burn_in < steps:
        raise ValueError("burn_in must lie between zero and steps")
    if lambda1 * mu > 1.0 or lambda2 * mu > 1.0:
        raise ValueError("lambda1*mu and lambda2*mu must not exceed one")

    master = np.random.default_rng(seed)
    a1 = power_law_quantiles(eta1, mean1, size, rng=master)
    a2 = power_law_quantiles(eta2, mean2, size, rng=master)
    master.shuffle(a2)
    mass = np.full(size, 1.0 / size)
    set_fraction = resolve_set_fraction(
        strategy, a1, a2, mass, immune_fraction, lambda1, lambda2, rho0
    )
    q = immunization_probabilities(
        strategy, a1, a2, mass, set_fraction, lambda1, lambda2, rho0
    )

    prevalence, actual_immune = [], []
    for _ in range(int(repeats)):
        rng = np.random.default_rng(master.integers(0, 2**32 - 1))
        p, immune = _single_run(
            a1, a2, q, set_fraction, strategy, observation_steps,
            lambda1 * mu, lambda2 * mu, mu, initial_prevalence,
            steps, burn_in, rng
        )
        prevalence.append(p)
        actual_immune.append(immune)
    return {
        "strategy": strategy,
        "set_fraction": float(set_fraction),
        "prevalence": float(np.mean(prevalence)),
        "prevalence_std": float(np.std(prevalence)),
        "immune_fraction": float(np.mean(actual_immune)),
        "repeats": int(repeats),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="R", choices=list(STRATEGIES))
    parser.add_argument("--size", type=int, default=1200)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--observation-steps", type=int, default=300)
    parser.add_argument("--immune-fraction", type=float, default=0.1)
    parser.add_argument("--lambda1", type=float, default=3.4)
    parser.add_argument("--lambda2", type=float, default=60.0)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--rho0", type=float, default=0.2)
    parser.add_argument("--initial-prevalence", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = run_simulation(
        strategy=args.strategy,
        size=args.size,
        steps=args.steps,
        burn_in=args.burn_in,
        repeats=args.repeats,
        observation_steps=args.observation_steps,
        immune_fraction=args.immune_fraction,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mu=args.mu,
        rho0=args.rho0,
        initial_prevalence=args.initial_prevalence,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
