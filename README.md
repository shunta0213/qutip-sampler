# qutip-sampler

[![Tests](https://github.com/shunta0213/qutip-sampler/actions/workflows/test.yml/badge.svg)](https://github.com/shunta0213/qutip-sampler/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/shunta0213/qutip-sampler/graph/badge.svg?token=U0SP82EUFP)](https://codecov.io/gh/shunta0213/qutip-sampler)
[![PyPI](https://img.shields.io/pypi/v/qutip-sampler)](https://pypi.org/project/qutip-sampler/)
[![Python](https://img.shields.io/pypi/pyversions/qutip-sampler)](https://pypi.org/project/qutip-sampler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A [dimod](https://github.com/dwavesystems/dimod)-compatible quantum annealing sampler backed by [QuTiP](https://qutip.org/) `sesolve`.

It maps an Ising / QUBO problem onto a transverse-field Ising Hamiltonian, evolves the ground state via a linear annealing schedule, and draws bitstring samples from the final quantum state.

## Installation

```bash
pip install qutip-sampler
```

## Quick start

```python
import dimod
from qutip_sampler import QuTipSampler

# SPIN (Ising) problem
bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': -1.0}, {('a', 'b'): 0.5}, 0.0, 'SPIN')
result = QuTipSampler().sample(bqm, num_reads=100, seed=42)
print(result.first.sample)   # e.g. {'a': 1, 'b': 1}

# Ising directly
result = QuTipSampler().sample_ising({'a': -1.0}, {('a', 'b'): 0.5}, num_reads=50)

# QUBO
Q = {('x', 'x'): -1.0, ('y', 'y'): -1.0, ('x', 'y'): 0.5}
result = QuTipSampler().sample_qubo(Q, num_reads=50, seed=0)
print(result.first.sample)   # e.g. {'x': 1, 'y': 1}
```

## Parameters

| Parameter     | Default | Description                                            |
| ------------- | ------- | ------------------------------------------------------ |
| `anneal_time` | `10.0`  | Total annealing time _T_ passed to `sesolve`           |
| `n_steps`     | `200`   | Number of time steps in the annealing schedule         |
| `num_reads`   | `100`   | Number of bitstring samples drawn from the final state |
| `seed`        | `None`  | RNG seed for reproducible sampling                     |

`anneal_time` and `n_steps` can be set at construction time or overridden per call.

## How it works

1. Collect all variables from the problem and sort them.
2. Build the Ising Hamiltonian `H_ising = Σ hᵢ σᵢᶻ + Σ Jᵢⱼ σᵢᶻ σⱼᶻ`.
3. Build the transverse-field Hamiltonian `H_T = −Σ σᵢˣ`, whose ground state is `|+⟩⊗ⁿ`.
4. Evolve `H(t) = (1 − t/T) H_T + (t/T) H_ising` from `t=0` to `t=T` using QuTiP `sesolve`.
5. Sample bitstrings from the probability distribution `|ψ(T)|²`.

## Release flow

Releases are fully automated via GitHub Actions:

1. **Every push / PR** — the [Tests](.github/workflows/test.yml) workflow runs the test suite and uploads coverage to Codecov.
2. **On a version tag** — the [Publish](.github/workflows/publish.yml) workflow runs tests first, then builds and publishes to PyPI if they pass.

To cut a release:

```bash
uvx bump-my-version bump patch   # or minor / major
git push && git push --tags
```

`bump-my-version` updates the version in `pyproject.toml`, commits, and creates the tag. Pushing the tag triggers the publish workflow.

## License

MIT
