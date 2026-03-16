"""
Quantum annealing sampler using QuTiP sesolve.

Maps a dimod BinaryQuadraticModel onto a transverse-field Ising model,
evolves the ground state of the transverse-field Hamiltonian to the Ising
Hamiltonian via a linear annealing schedule, then samples bitstrings from
the final quantum state probability distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import dimod
import qutip

__all__ = ["QuTipSampler"]


def _operator_on_qubit(op: qutip.Qobj, qubit_idx: int, n_qubits: int) -> qutip.Qobj:
    """Embed a single-qubit operator into the N-qubit Hilbert space.

    Builds: I x ... x op x ... x I  where op sits at position qubit_idx.
    """
    ops = [qutip.identity(2)] * n_qubits
    ops[qubit_idx] = op
    return qutip.tensor(ops)


def _build_ising_hamiltonian(
    h: dict[Any, float],
    J: dict[tuple[Any, Any], float],
    variables: list[Any],
) -> qutip.Qobj:
    """Build H_ising = sum_i h[i]*sigma_z^(i) + sum_(i,j) J[i,j]*sigma_z^(i)*sigma_z^(j)."""
    n = len(variables)
    var_to_idx = {v: i for i, v in enumerate(variables)}

    H = qutip.qzero([2] * n)

    for var, bias in h.items():
        if bias == 0.0:
            continue
        H += bias * _operator_on_qubit(qutip.sigmaz(), var_to_idx[var], n)

    for (var_i, var_j), coupling in J.items():
        if coupling == 0.0:
            continue
        ops = [qutip.identity(2)] * n
        ops[var_to_idx[var_i]] = qutip.sigmaz()
        ops[var_to_idx[var_j]] = qutip.sigmaz()
        H += coupling * qutip.tensor(ops)

    return H


def _build_transverse_hamiltonian(n_qubits: int) -> qutip.Qobj:
    """Build H_transverse = -sum_i sigma_x^(i).

    The negative sign ensures |+>^N is the ground state.
    """
    H = qutip.qzero([2] * n_qubits)
    for i in range(n_qubits):
        H -= _operator_on_qubit(qutip.sigmax(), i, n_qubits)
    return H


def _initial_state(n_qubits: int) -> qutip.Qobj:
    """Ground state of H_transverse: tensor product of |+> = (|0>+|1>)/sqrt(2)."""
    plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    return qutip.tensor([plus] * n_qubits)


def _sample_from_state(
    final_state: qutip.Qobj,
    variables: list[Any],
    num_reads: int,
    rng: np.random.Generator,
) -> list[dict[Any, int]]:
    """Draw bitstring samples from the final quantum state.

    Basis index i -> binary string -> qubit 0 is MSB.
    |0> (bit '0') -> sigma_z = +1 -> spin +1.
    |1> (bit '1') -> sigma_z = -1 -> spin -1.
    """
    n = len(variables)
    amplitudes = final_state.full().flatten()
    probs = np.abs(amplitudes) ** 2
    probs /= probs.sum()

    indices = rng.choice(2**n, size=num_reads, p=probs)

    return [
        {var: (+1 if bits[i] == "0" else -1) for i, var in enumerate(variables)}
        for idx in indices
        for bits in [format(idx, f"0{n}b")]
    ]


def _anneal(
    h: dict[Any, float],
    J: dict[tuple[Any, Any], float],
    variables: list[Any],
    T: float,
    steps: int,
) -> qutip.Qobj:
    """Run the quantum annealing schedule and return the final state."""
    n = len(variables)
    H_ising = _build_ising_hamiltonian(h, J, variables)
    H_transverse = _build_transverse_hamiltonian(n)
    psi0 = _initial_state(n)
    tlist = np.linspace(0.0, T, steps + 1)
    H_td = [
        [H_transverse, lambda t, _=None: 1.0 - t / T],
        [H_ising, lambda t, _=None: t / T],
    ]
    return qutip.sesolve(H_td, psi0, tlist, e_ops=[]).states[-1]


class QuTipSampler(dimod.Sampler):
    """dimod Sampler that simulates quantum annealing via QuTiP sesolve.

    Takes a dimod.BinaryQuadraticModel (via sample/sample_ising/sample_qubo),
    maps it to a transverse-field Ising model, evolves the state with a linear
    annealing schedule using sesolve, and returns a dimod.SampleSet.

    Parameters
    ----------
    anneal_time : float
        Total annealing time T (default 10.0).
    n_steps : int
        Number of time steps for sesolve (default 200).

    Example
    -------
    >>> import dimod
    >>> from qutip_sampler import QuTipSampler
    >>> bqm = dimod.BinaryQuadraticModel({'a': -1, 'b': -1}, {'ab': 0.5}, 0.0, 'SPIN')
    >>> result = QuTipSampler().sample(bqm, num_reads=100, seed=42)
    >>> print(result.first.sample)
    """

    def __init__(self, anneal_time: float = 10.0, n_steps: int = 200) -> None:
        self._anneal_time = anneal_time
        self._n_steps = n_steps
        self._parameters: dict[str, list] = {
            "num_reads": [],
            "seed": [],
            "anneal_time": [],
            "n_steps": [],
        }
        self._properties: dict[str, Any] = {
            "description": "Quantum annealing sampler backed by QuTiP sesolve",
            "annealing_schedule": "linear",
        }

    @property
    def parameters(self) -> dict[str, list]:
        return self._parameters

    @property
    def properties(self) -> dict[str, Any]:
        return self._properties

    def _resolve(
        self, anneal_time: float | None, n_steps: int | None
    ) -> tuple[float, int]:
        """Return effective (T, steps), falling back to instance defaults."""
        return (
            anneal_time if anneal_time is not None else self._anneal_time,
            n_steps if n_steps is not None else self._n_steps,
        )

    def sample_ising(
        self,
        h: dict[Any, float],
        J: dict[tuple[Any, Any], float],
        *,
        num_reads: int = 100,
        seed: int | None = None,
        anneal_time: float | None = None,
        n_steps: int | None = None,
        **kwargs,
    ) -> dimod.SampleSet:
        """Sample from an Ising problem via quantum annealing simulation.

        Parameters
        ----------
        h : dict
            Linear biases {variable: bias}.
        J : dict
            Quadratic couplings {(var_i, var_j): coupling}.
        num_reads : int
            Number of bitstring samples to draw (default 100).
        seed : int or None
            RNG seed for reproducible sampling.
        anneal_time : float or None
            Override the instance anneal_time.
        n_steps : int or None
            Override the instance n_steps.

        Returns
        -------
        dimod.SampleSet
            Samples with SPIN vartype and corresponding Ising energies.
        """
        all_vars: set[Any] = set(h.keys())
        for vi, vj in J.keys():
            all_vars.update([vi, vj])
        variables = sorted(all_vars)

        if not variables:
            return dimod.SampleSet.from_samples([], dimod.SPIN, energy=[])

        T, steps = self._resolve(anneal_time, n_steps)
        final_state = _anneal(h, J, variables, T, steps)

        rng = np.random.default_rng(seed)
        raw_samples = _sample_from_state(final_state, variables, num_reads, rng)
        energies = [dimod.ising_energy(s, h, J) for s in raw_samples]

        return dimod.SampleSet.from_samples(raw_samples, vartype=dimod.SPIN, energy=energies)

    def sample_qubo(
        self,
        Q: dict[tuple[Any, Any], float],
        *,
        num_reads: int = 100,
        seed: int | None = None,
        anneal_time: float | None = None,
        n_steps: int | None = None,
        **kwargs,
    ) -> dimod.SampleSet:
        """Sample from a QUBO problem via quantum annealing simulation.

        Parameters
        ----------
        Q : dict
            QUBO coefficients {(var_i, var_j): value}. Diagonal entries
            (var, var) are linear biases; off-diagonal are quadratic.
        num_reads : int
            Number of bitstring samples to draw (default 100).
        seed : int or None
            RNG seed for reproducible sampling.
        anneal_time : float or None
            Override the instance anneal_time.
        n_steps : int or None
            Override the instance n_steps.

        Returns
        -------
        dimod.SampleSet
            Samples with BINARY vartype and corresponding QUBO energies.
        """
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        return self.sample(bqm, num_reads=num_reads, seed=seed, anneal_time=anneal_time, n_steps=n_steps, **kwargs)

    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        *,
        num_reads: int = 100,
        seed: int | None = None,
        anneal_time: float | None = None,
        n_steps: int | None = None,
        **kwargs,
    ) -> dimod.SampleSet:
        """Sample from a BinaryQuadraticModel via quantum annealing simulation.

        Parameters
        ----------
        bqm : dimod.BinaryQuadraticModel
            The problem to sample. Supports both SPIN and BINARY vartypes.
        num_reads : int
            Number of bitstring samples to draw (default 100).
        seed : int or None
            RNG seed for reproducible sampling.
        anneal_time : float or None
            Override the instance anneal_time.
        n_steps : int or None
            Override the instance n_steps.

        Returns
        -------
        dimod.SampleSet
            Samples with the same vartype as the input BQM.
        """
        h, J, offset = bqm.to_ising()
        sampleset = self.sample_ising(
            h, J,
            num_reads=num_reads,
            seed=seed,
            anneal_time=anneal_time,
            n_steps=n_steps,
            **kwargs,
        )
        return sampleset.change_vartype(bqm.vartype, energy_offset=offset)
