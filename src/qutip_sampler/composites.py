"""
QPU auto-scaling composite for QuTipSampler.

Emulates the D-Wave QPU ``auto_scale`` parameter:
https://docs.dwavequantum.com/en/latest/quantum_research/solver_parameters.html#parameter-qpu-auto-scale

The scalar is chosen so that h and J fit within the QPU's hardware ranges.
Both h and J are divided by the scalar before annealing, and energies are
rescaled back afterwards so the returned SampleSet reflects the original problem.
"""

from __future__ import annotations

from typing import Any

import dimod
from dwave.system.coupling_groups import coupling_groups

__all__ = ["QPUAutoScaleComposite"]

# Advantage2 QPU hardware limits (default).
# Advantage:  h_range=(-2, 2), j_range=(-1, 1), coupling_range=(-18, 15)
# Advantage2: h_range=(-4, 4), j_range=(-2, 1), coupling_range=(-13, 10)
_DEFAULT_H_RANGE: tuple[float, float] = (-4.0, 4.0)
_DEFAULT_J_RANGE: tuple[float, float] = (-2.0, 1.0)
_DEFAULT_COUPLING_RANGE: tuple[float, float] = (-13.0, 10.0)


def _coupling_limit(
    J: dict[tuple[Any, Any], float],
    coupling_range: tuple[float, float],
    hardware_graph,
) -> float:
    """Compute the coupling_limit term of the auto-scale scalar.

    For Zephyr (Advantage2) graphs, uses ``dwave.system.coupling_groups`` to
    obtain the exact per-group coupler totals, matching the QPU's
    ``per_group_coupling_range`` constraint.

    For non-Zephyr topologies (Pegasus / Chimera), falls back to per-qubit
    totals as a conservative upper bound.

    Parameters
    ----------
    J : dict
        Quadratic couplings {(vi, vj): value}.
    coupling_range : (float, float)
        Supported (min, max) range for per-group (or per-qubit) total coupling.
    hardware_graph : networkx.Graph
        QPU hardware graph from ``DWaveSampler.to_networkx_graph()``.

    Returns
    -------
    float
        The coupling_limit term (>= 0.0).
    """
    if not J:
        return 0.0

    if not hardware_graph.graph.get("family") == "zephyr":
        raise ValueError(
            "QPUAutoScaleComposite only supports Zephyr topology, got %s"
            % hardware_graph.graph.get("family")
        )

    # Exact per-group totals for Zephyr (Advantage2) topology.
    group_totals = []
    for group in coupling_groups(hardware_graph):
        total = sum(J.get((u, v), J.get((v, u), 0.0)) for u, v in group)
        if total != 0.0:
            group_totals.append(total)

    if not group_totals:
        return 0.0

    return max(
        max(max(group_totals) / coupling_range[1], 0.0),
        max(min(group_totals) / coupling_range[0], 0.0),
    )


def _compute_scalar(
    h: dict[Any, float],
    J: dict[tuple[Any, Any], float],
    h_range: tuple[float, float],
    j_range: tuple[float, float],
    coupling_range: tuple[float, float],
    hardware_graph,
) -> float:
    """Compute the auto-scale scalar following the D-Wave QPU algorithm.

    scalar = max(
        max(h) / h_range[1],
        min(h) / h_range[0],
        max(J) / j_range[1],
        min(J) / j_range[0],
        coupling_limit,
    )

    Each term is clamped to 0 from below so only range violations contribute.
    The result is at least 1.0 (values are never amplified).

    Parameters
    ----------
    h : dict
        Linear biases.
    J : dict
        Quadratic couplings.
    h_range : (float, float)
        Supported (min, max) range for linear biases.
    j_range : (float, float)
        Supported (min, max) range for quadratic couplings.
    coupling_range : (float, float)
        Supported (min, max) range for per-group (or per-qubit) total coupling.
    hardware_graph : networkx.Graph
        QPU hardware graph from ``DWaveSampler.to_networkx_graph()``.

    Returns
    -------
    float
        The scalar divisor (>= 1.0).
    """
    h_vals = list(h.values()) if h else [0.0]
    J_vals = list(J.values()) if J else [0.0]

    candidates = [
        max(max(h_vals) / h_range[1], 0.0),
        max(min(h_vals) / h_range[0], 0.0),
        max(max(J_vals) / j_range[1], 0.0),
        max(min(J_vals) / j_range[0], 0.0),
        _coupling_limit(J, coupling_range, hardware_graph),
    ]

    return max(max(candidates), 1.0)


class QPUAutoScaleComposite(dimod.ComposedSampler):
    """Composite that emulates the D-Wave QPU ``auto_scale`` feature.

    Scales h and J so they fit within the QPU's hardware ranges before passing
    them to the child sampler, then rescales energies back to the original
    problem. The scalar is the minimum divisor that brings all values within
    range.

    Defaults to Advantage2 QPU hardware ranges. Pass ``h_range``, ``j_range``,
    and ``coupling_range`` to emulate a different solver architecture
    (e.g. Advantage: h_range=(-2, 2), j_range=(-1, 1), coupling_range=(-18, 15)).

    Parameters
    ----------
    child_sampler : dimod.Sampler
        The sampler to wrap (typically :class:`~qutip_sampler.QuTipSampler`).
    hardware_graph : networkx.Graph
        QPU hardware graph from ``DWaveSampler.to_networkx_graph()``.
        For Zephyr (Advantage2) graphs, exact per-group coupling totals are used.
        For Pegasus / Chimera graphs, per-qubit totals are used instead.
    h_range : (float, float)
        Supported range for linear biases. Default: (-4.0, 4.0).
    j_range : (float, float)
        Supported range for quadratic couplings. Default: (-2.0, 1.0).
    coupling_range : (float, float)
        Supported range for per-group (or per-qubit) total coupling.
        Default: (-13.0, 10.0).

    Example
    -------
    >>> from dwave.system import DWaveSampler
    >>> from qutip_sampler import QuTipSampler
    >>> from qutip_sampler.composites import QPUAutoScaleComposite
    >>> qpu = DWaveSampler(solver=dict(topology__type='zephyr'))
    >>> sampler = QPUAutoScaleComposite(
    ...     QuTipSampler(),
    ...     hardware_graph=qpu.to_networkx_graph(),
    ... )
    >>> result = sampler.sample_ising({'a': -8.0, 'b': 6.0}, {}, num_reads=10)
    >>> result.info['scalar']   # 8.0 / 4.0 = 2.0
    2.0
    """

    def __init__(
        self,
        child_sampler: dimod.Sampler,
        hardware_graph,
        *,
        h_range: tuple[float, float] = _DEFAULT_H_RANGE,
        j_range: tuple[float, float] = _DEFAULT_J_RANGE,
        coupling_range: tuple[float, float] = _DEFAULT_COUPLING_RANGE,
    ) -> None:
        self._children = [child_sampler]
        self._h_range = h_range
        self._j_range = j_range
        self._coupling_range = coupling_range
        self._hardware_graph = hardware_graph

    @property
    def children(self) -> list[dimod.Sampler]:
        return self._children

    @property
    def parameters(self) -> dict[str, list]:
        params = self.child.parameters.copy()
        params["auto_scale"] = []
        return params

    @property
    def properties(self) -> dict[str, Any]:
        topology = (
            self._hardware_graph.graph.get("family", "unknown")
            if self._hardware_graph is not None
            else None
        )
        return {
            "child_properties": self.child.properties.copy(),
            "h_range": list(self._h_range),
            "j_range": list(self._j_range),
            "coupling_range": list(self._coupling_range),
            "hardware_topology": topology,
        }

    def sample_ising(
        self,
        h: dict[Any, float],
        J: dict[tuple[Any, Any], float],
        *,
        auto_scale: bool = True,
        **parameters,
    ) -> dimod.SampleSet:
        """Sample from an Ising problem with optional QPU-style auto-scaling.

        Parameters
        ----------
        h : dict
            Linear biases.
        J : dict
            Quadratic couplings.
        auto_scale : bool
            When True (default), h and J are divided by the computed scalar
            before sampling and energies are restored afterwards. When False,
            h and J are passed through unchanged.
        **parameters
            Forwarded to the child sampler.

        Returns
        -------
        dimod.SampleSet
            Samples in SPIN vartype. ``sampleset.info['scalar']`` contains
            the scalar that was applied (1.0 when auto_scale=False).
        """
        if not auto_scale:
            sampleset = self.child.sample_ising(h, J, **parameters)
            sampleset.info["scalar"] = 1.0
            return sampleset

        scalar = _compute_scalar(
            h,
            J,
            self._h_range,
            self._j_range,
            self._coupling_range,
            self._hardware_graph,
        )

        h_scaled = {v: bias / scalar for v, bias in h.items()}
        J_scaled = {edge: coupling / scalar for edge, coupling in J.items()}

        sampleset = self.child.sample_ising(h_scaled, J_scaled, **parameters)

        sampleset.record.energy *= scalar
        sampleset.info["scalar"] = scalar

        return sampleset

    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        *,
        auto_scale: bool = True,
        **parameters,
    ) -> dimod.SampleSet:
        """Sample from a BQM with optional QPU-style auto-scaling."""
        h, J, offset = bqm.to_ising()
        sampleset = self.sample_ising(h, J, auto_scale=auto_scale, **parameters)
        return sampleset.change_vartype(bqm.vartype, energy_offset=offset)

    def sample_qubo(
        self,
        Q: dict[tuple[Any, Any], float],
        *,
        auto_scale: bool = True,
        **parameters,
    ) -> dimod.SampleSet:
        """Sample from a QUBO with optional QPU-style auto-scaling."""
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        return self.sample(bqm, auto_scale=auto_scale, **parameters)
