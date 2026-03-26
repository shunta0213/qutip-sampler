"""Tests for qutip_sampler.composites (QPUAutoScaleComposite)."""

from __future__ import annotations

import dimod
import dwave_networkx as dnx
import pytest
from dwave.system.coupling_groups import coupling_groups

from qutip_sampler import QPUAutoScaleComposite, QuTipSampler
from qutip_sampler.composites import _compute_scalar, _coupling_limit

# Advantage2 defaults used throughout unless overridden.
H_RANGE = (-4.0, 4.0)
J_RANGE = (-2.0, 1.0)
COUPLING_RANGE = (-13.0, 10.0)

# Small synthetic hardware graphs shared across tests.
ZEPHYR_GRAPH = dnx.zephyr_graph(2)  # Advantage2 topology
PEGASUS_GRAPH = dnx.pegasus_graph(2)  # Advantage topology (non-Zephyr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zephyr_edges(n: int) -> dict:
    """Return the first n edges of ZEPHYR_GRAPH as a J dict with value 0.5."""
    return {e: 0.5 for e in list(ZEPHYR_GRAPH.edges())[:n]}


def _first_zephyr_groups(qubit):
    """Return (group0_edges, group1_edges) for the given qubit in ZEPHYR_GRAPH."""
    qubit_groups = [
        g for g in coupling_groups(ZEPHYR_GRAPH) if any(qubit in edge for edge in g)
    ]
    return qubit_groups[0], qubit_groups[1]


# ---------------------------------------------------------------------------
# Unit tests: _coupling_limit
# ---------------------------------------------------------------------------


class TestCouplingLimit:
    def test_empty_J_returns_zero(self):
        assert _coupling_limit({}, COUPLING_RANGE, ZEPHYR_GRAPH) == 0.0

    def test_zephyr_within_range_returns_non_negative(self):
        """Small J values well within coupling_range → limit is >= 0 and < 1."""
        J = _zephyr_edges(1)  # single edge, tiny group total
        limit = _coupling_limit(J, COUPLING_RANGE, ZEPHYR_GRAPH)
        assert 0.0 <= limit < 1.0

    def test_zephyr_per_group_exceeds_max(self):
        """Group total > coupling_range[1] → limit > 0."""
        g0, _ = _first_zephyr_groups(0)
        # Fill group0 with 1.8 each: total = 1.8 * len(g0) > 10.0
        n = len(g0)
        J = {(min(u, v), max(u, v)): 1.8 for u, v in g0}
        expected = 1.8 * n / COUPLING_RANGE[1]
        limit = _coupling_limit(J, COUPLING_RANGE, ZEPHYR_GRAPH)
        assert abs(limit - expected) < 1e-10

    def test_zephyr_per_group_exceeds_min(self):
        """Group total < coupling_range[0] → limit > 0."""
        g0, _ = _first_zephyr_groups(0)
        n = len(g0)
        J = {(min(u, v), max(u, v)): -2.2 for u, v in g0}
        expected = 2.2 * n / abs(COUPLING_RANGE[0])
        limit = _coupling_limit(J, COUPLING_RANGE, ZEPHYR_GRAPH)
        assert abs(limit - expected) < 1e-10

    def test_zephyr_cancellation_across_groups_detected(self):
        """Couplings cancel per-qubit but violate per-group range.

        group0 total = +large, group1 total = -large.
        Per-qubit sum ≈ 0, but each group individually violates the range.
        Only the per-group path detects this; per-qubit path would miss it.
        """
        g0, g1 = _first_zephyr_groups(0)
        n0, n1 = len(g0), len(g1)
        # Push group0 to +12 (> coupling_range[1]=10) and group1 to -14 (< -13)
        J = {}
        for u, v in g0:
            J[(min(u, v), max(u, v))] = 12.0 / n0
        for u, v in g1:
            J[(min(u, v), max(u, v))] = -14.0 / n1

        limit_zephyr = _coupling_limit(J, COUPLING_RANGE, ZEPHYR_GRAPH)
        # Binding group: group1 total = -14 → 14/13 ≈ 1.077
        assert limit_zephyr > 1.0


# ---------------------------------------------------------------------------
# Unit tests: _compute_scalar
# ---------------------------------------------------------------------------


class TestComputeScalar:
    def test_within_range_returns_one(self):
        h = {"a": 1.0, "b": -1.0}
        J = _zephyr_edges(1)
        assert (
            _compute_scalar(h, J, H_RANGE, J_RANGE, COUPLING_RANGE, ZEPHYR_GRAPH) == 1.0
        )

    def test_h_exceeds_max(self):
        """h=8.0, h_range[1]=4.0 → scalar=2.0."""
        h = {"a": 8.0}
        assert (
            _compute_scalar(h, {}, H_RANGE, J_RANGE, COUPLING_RANGE, ZEPHYR_GRAPH)
            == 2.0
        )

    def test_h_exceeds_min(self):
        """h=-8.0, h_range[0]=-4.0 → scalar=2.0."""
        h = {"a": -8.0}
        assert (
            _compute_scalar(h, {}, H_RANGE, J_RANGE, COUPLING_RANGE, ZEPHYR_GRAPH)
            == 2.0
        )

    def test_J_exceeds_max(self):
        """J=3.0, j_range[1]=1.0 → scalar=3.0 (independent of topology)."""
        J = {list(ZEPHYR_GRAPH.edges())[0]: 3.0}
        assert (
            _compute_scalar({}, J, H_RANGE, J_RANGE, COUPLING_RANGE, ZEPHYR_GRAPH)
            == 3.0
        )

    def test_J_exceeds_min(self):
        """J=-6.0, j_range[0]=-2.0 → scalar=3.0."""
        J = {list(ZEPHYR_GRAPH.edges())[0]: -6.0}
        assert (
            _compute_scalar({}, J, H_RANGE, J_RANGE, COUPLING_RANGE, ZEPHYR_GRAPH)
            == 3.0
        )

    def test_largest_violation_wins(self):
        h = {"a": 8.0}  # 8/4 = 2.0
        J = {list(ZEPHYR_GRAPH.edges())[0]: 3.0}  # 3/1 = 3.0 ← largest
        scalar = _compute_scalar(h, J, H_RANGE, J_RANGE, COUPLING_RANGE, ZEPHYR_GRAPH)
        assert abs(scalar - 3.0) < 1e-10

    def test_empty_h_and_J(self):
        assert (
            _compute_scalar({}, {}, H_RANGE, J_RANGE, COUPLING_RANGE, ZEPHYR_GRAPH)
            == 1.0
        )

    def test_custom_advantage_ranges(self):
        """Custom h_range=(-2,2): h=3.0, h_range[1]=2.0 → scalar=1.5."""
        h = {"a": 3.0}
        scalar = _compute_scalar(
            h, {}, (-2.0, 2.0), (-1.0, 1.0), (-18.0, 15.0), ZEPHYR_GRAPH
        )
        assert abs(scalar - 1.5) < 1e-10

    def test_documentation_worked_example(self):
        """Verbatim example from the D-Wave auto_scale docs.

        h={'a': -7.2, 'b': 2.3}, J={('a','b'): 1.5}
        Advantage2: h_range=(-4,4), extended_j_range=(-2,1)
        Binding: min(h)/min(h_range) = -7.2/-4.0 = 1.8
        """
        h = {"a": -7.2, "b": 2.3}
        edge = list(ZEPHYR_GRAPH.edges())[0]
        J = {edge: 1.5}
        scalar = _compute_scalar(
            h, J, (-4.0, 4.0), (-2.0, 1.0), (-13.0, 10.0), ZEPHYR_GRAPH
        )
        assert abs(scalar - 1.8) < 1e-10


# ---------------------------------------------------------------------------
# Integration tests: QPUAutoScaleComposite
# ---------------------------------------------------------------------------


class TestQPUAutoScaleComposite:
    def setup_method(self):
        self.sampler = QPUAutoScaleComposite(QuTipSampler(), ZEPHYR_GRAPH)

    # --- dimod interface ---

    def test_is_dimod_sampler(self):
        assert isinstance(self.sampler, dimod.Sampler)

    def test_parameters_contains_auto_scale(self):
        assert "auto_scale" in self.sampler.parameters

    def test_properties_contains_ranges(self):
        props = self.sampler.properties
        assert props["h_range"] == list(H_RANGE)
        assert props["j_range"] == list(J_RANGE)
        assert props["coupling_range"] == list(COUPLING_RANGE)

    def test_properties_hardware_topology_zephyr(self):
        assert self.sampler.properties["hardware_topology"] == "zephyr"

    def test_non_zephyr_graph_raises(self):
        """Passing a non-Zephyr graph must raise ValueError on sample."""
        sampler = QPUAutoScaleComposite(QuTipSampler(), PEGASUS_GRAPH)
        with pytest.raises(ValueError, match="Zephyr"):
            sampler.sample_ising({"a": -1.0}, {("a", "b"): 0.5}, num_reads=1)

    # --- scalar stored in info ---

    def test_scalar_stored_in_info_when_scaling(self):
        h = {"a": -8.0}  # exceeds h_range[1]=4.0 → scalar=2.0
        result = self.sampler.sample_ising(h, {}, num_reads=5, seed=0)
        assert abs(result.info["scalar"] - 2.0) < 1e-10

    def test_scalar_is_one_when_within_range(self):
        h = {"a": -1.0, "b": 1.0}
        result = self.sampler.sample_ising(h, {}, num_reads=5, seed=0)
        assert result.info["scalar"] == 1.0

    def test_scalar_is_one_when_auto_scale_false(self):
        h = {"a": -8.0}
        result = self.sampler.sample_ising(h, {}, num_reads=5, seed=0, auto_scale=False)
        assert result.info["scalar"] == 1.0

    # --- energy correctness ---

    def test_energies_match_original_problem_ising(self):
        h = {"a": -8.0, "b": 4.0}
        J = {("a", "b"): -3.0}
        result = self.sampler.sample_ising(h, J, num_reads=20, seed=0)
        for sample, energy in result.data(["sample", "energy"]):
            assert abs(energy - dimod.ising_energy(sample, h, J)) < 1e-6

    def test_energies_match_original_problem_no_scaling(self):
        h = {"a": -1.0, "b": 0.5}
        J = {("a", "b"): -0.5}
        result = self.sampler.sample_ising(h, J, num_reads=20, seed=0)
        for sample, energy in result.data(["sample", "energy"]):
            assert abs(energy - dimod.ising_energy(sample, h, J)) < 1e-6

    # --- vartype ---

    def test_vartype_spin_for_ising(self):
        result = self.sampler.sample_ising({"a": -1.0}, {}, num_reads=5, seed=0)
        assert result.vartype == dimod.SPIN

    def test_vartype_preserved_for_spin_bqm(self):
        bqm = dimod.BinaryQuadraticModel({"a": -8.0}, {}, 0.0, "SPIN")
        result = self.sampler.sample(bqm, num_reads=5, seed=0)
        assert result.vartype == dimod.SPIN

    def test_vartype_preserved_for_binary_bqm(self):
        bqm = dimod.BinaryQuadraticModel({"a": -8.0}, {}, 0.0, "BINARY")
        result = self.sampler.sample(bqm, num_reads=5, seed=0)
        assert result.vartype == dimod.BINARY

    def test_vartype_binary_for_qubo(self):
        Q = {("a", "a"): -8.0, ("b", "b"): -8.0, ("a", "b"): 3.0}
        result = self.sampler.sample_qubo(Q, num_reads=5, seed=0)
        assert result.vartype == dimod.BINARY

    # --- sample / sample_qubo ---

    def test_sample_qubo_energies_match_original(self):
        Q = {("a", "a"): -8.0, ("b", "b"): -8.0, ("a", "b"): 3.0}
        result = self.sampler.sample_qubo(Q, num_reads=20, seed=0)
        for sample, energy in result.data(["sample", "energy"]):
            assert abs(energy - dimod.qubo_energy(sample, Q)) < 1e-6

    def test_sample_bqm_energies_match_original(self):
        bqm = dimod.BinaryQuadraticModel(
            {"a": -8.0, "b": 4.0}, {("a", "b"): -3.0}, 0.0, "SPIN"
        )
        result = self.sampler.sample(bqm, num_reads=20, seed=0)
        for sample, energy in result.data(["sample", "energy"]):
            assert abs(energy - bqm.energy(sample)) < 1e-6

    # --- custom ranges with Advantage (Pegasus) ---

    def test_custom_ranges_scalar(self):
        """Custom h_range=(-2,2): h=3.0 exceeds h_range[1]=2.0 → scalar=1.5."""
        sampler = QPUAutoScaleComposite(
            QuTipSampler(),
            ZEPHYR_GRAPH,
            h_range=(-2.0, 2.0),
            j_range=(-1.0, 1.0),
            coupling_range=(-13.0, 10.0),
        )
        result = sampler.sample_ising({"a": 3.0}, {}, num_reads=5, seed=0)
        assert abs(result.info["scalar"] - 1.5) < 1e-10

    def test_custom_ranges_reflected_in_properties(self):
        sampler = QPUAutoScaleComposite(
            QuTipSampler(),
            ZEPHYR_GRAPH,
            h_range=(-2.0, 2.0),
            j_range=(-1.0, 1.0),
            coupling_range=(-13.0, 10.0),
        )
        assert sampler.properties["h_range"] == [-2.0, 2.0]
        assert sampler.properties["j_range"] == [-1.0, 1.0]
        assert sampler.properties["coupling_range"] == [-13.0, 10.0]
