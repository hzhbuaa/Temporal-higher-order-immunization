# Immunization on Temporal Higher-Order Networks

This repository contains the theory and stochastic simulation code used to study 
immunization strategies on higher-order temporal networks. The associated paper, 
*Immunization on Temporal Higher-Order Networks*, is available at [arXiv:2607.10171](https://arxiv.org/abs/2607.10171).

## Files

- `theory.py`: stationary mean-field equations and numerical solution.
- `simulation.py`: stochastic temporal-network simulation for a given parameter set.
- `utils.py`: power-law activity distributions and immunization strategies.

The implemented strategies are:

- `R`: random immunization.
- `HIC`: ranking by the higher-order infection kernel.
- `TA`: ranking by `a1 + a2`.
- `HA`: ranking by higher-order activity `a2`.
- `PA`: ranking by pairwise activity `a1`.
- `EHS`: higher-order egocentric sampling.
- `EPS`: pairwise egocentric sampling.
- `EBS`: combined egocentric sampling.

For `EPS`, `EHS`, and `EBS`, the simulation samples ego nodes, observes temporal
contacts, and selects neighbors with their observed contact counts as weights. The
theory uses the corresponding large-network immunization probabilities.

## Requirements

- Python 3.8 or later
- NumPy
- SciPy

Install the dependency with:

```bash
python -m pip install numpy scipy
```

## Usage

Run the theory for one strategy:

```bash
python theory.py --strategy HIC
```

Run all theoretical strategies:

```bash
python theory.py --strategy all
```

Find all distinct equilibria from multiple initial guesses and report the
pre-immunization prevalence mapped from each equilibrium using Eqs. (22)-(23) of
the paper:

```bash
python theory.py --strategy R --immune-fraction 0.33 --all-equilibria
```

Increase `--initial-guesses` when resolving closely spaced equilibrium branches.
The Python function `find_equilibria` also returns the post-immunization and mapped
pre-immunization infection profiles for every equilibrium.

`--immune-fraction` always denotes the actual immunized population fraction
\(\omega\). For `EPS`, `EHS`, and `EBS`, the programs first determine the probe
fraction \(\phi\) that produces the requested \(\omega\); this value is reported as
`set_fraction`.

Run a stochastic simulation:

```bash
python simulation.py --strategy HIC
```

The principal parameters, including network size, infection strengths, immune
fraction, run length, and random seed, are available as command-line options. Use
`python theory.py --help` or `python simulation.py --help` for the complete list.
