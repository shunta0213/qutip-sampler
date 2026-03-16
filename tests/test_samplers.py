"""Tests for qutip_sampler.samplers."""
from __future__ import annotations

import numpy as np
import pytest
import dimod

from dwave.preprocessing.composites import ScaleComposite, SpinReversalTransformComposite

from qutip_sampler import QuTipSampler
from qutip_sampler.samplers import (
    _build_ising_hamiltonian,
    _build_transverse_hamiltonian,
    _initial_state,
    _operator_on_qubit,
)


# ---------------------------------------------------------------------------
# Unit tests: Hamiltonian helpers
# ---------------------------------------------------------------------------

class TestOperatorOnQubit:
    def test_single_qubit(self):
        import qutip
        op = _operator_on_qubit(qutip.sigmaz(), 0, 1)
        assert (op - qutip.sigmaz()).norm() < 1e-10

    def test_two_qubit_first(self):
        import qutip
        op = _operator_on_qubit(qutip.sigmaz(), 0, 2)
        expected = qutip.tensor(qutip.sigmaz(), qutip.identity(2))
        assert (op - expected).norm() < 1e-10

    def test_two_qubit_second(self):
        import qutip
        op = _operator_on_qubit(qutip.sigmaz(), 1, 2)
        expected = qutip.tensor(qutip.identity(2), qutip.sigmaz())
        assert (op - expected).norm() < 1e-10


class TestIsinhamiltonianConstruction:
    def test_single_qubit_linear(self):
        import qutip
        H = _build_ising_hamiltonian({'a': 2.0}, {}, ['a'])
        assert (H - 2.0 * qutip.sigmaz()).norm() < 1e-10

    def test_two_qubit_coupling(self):
        import qutip
        H = _build_ising_hamiltonian({}, {('a', 'b'): 0.5}, ['a', 'b'])
        expected = 0.5 * qutip.tensor(qutip.sigmaz(), qutip.sigmaz())
        assert (H - expected).norm() < 1e-10

    def test_zero_coupling_skipped(self):
        """J coupling of 0.0 must be a no-op (exercises the continue branch)."""
        H_zero_coupling = _build_ising_hamiltonian({}, {('a', 'b'): 0.0}, ['a', 'b'])
        H_no_coupling = _build_ising_hamiltonian({}, {}, ['a', 'b'])
        assert (H_zero_coupling - H_no_coupling).norm() < 1e-10


class TestTransverseHamiltonian:
    def test_single_qubit(self):
        import qutip
        H = _build_transverse_hamiltonian(1)
        assert (H + qutip.sigmax()).norm() < 1e-10

    def test_ground_state_energy(self):
        """Ground state energy of -sum sigma_x^(i) should be -N."""
        for n in [1, 2, 3]:
            H = _build_transverse_hamiltonian(n)
            assert abs(min(H.eigenenergies()) + n) < 1e-8


class TestInitialState:
    def test_is_normalised(self):
        psi = _initial_state(3)
        assert abs(psi.norm() - 1.0) < 1e-10

    def test_uniform_probabilities(self):
        n = 3
        psi = _initial_state(n)
        probs = np.abs(psi.full().flatten()) ** 2
        assert np.allclose(probs, 1.0 / 2**n, atol=1e-10)

    def test_is_ground_state_of_transverse(self):
        import qutip
        for n in [1, 2, 3]:
            H = _build_transverse_hamiltonian(n)
            psi = _initial_state(n)
            eigenvalues, eigenstates = H.eigenstates()
            overlap = abs(eigenstates[0].dag().overlap(psi))
            assert abs(overlap - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# Integration tests: dimod Sampler interface
# ---------------------------------------------------------------------------

class TestSamplerInterface:
    def setup_method(self):
        self.sampler = QuTipSampler()

    def test_is_dimod_sampler(self):
        assert isinstance(self.sampler, dimod.Sampler)

    def test_parameters_contains_expected_keys(self):
        params = self.sampler.parameters
        assert set(params) == {'num_reads', 'seed', 'anneal_time', 'n_steps'}

    def test_properties_contains_expected_keys(self):
        props = self.sampler.properties
        assert 'description' in props
        assert 'annealing_schedule' in props

    def test_sample_ising_returns_sampleset(self):
        result = self.sampler.sample_ising({'a': -1.0}, {}, num_reads=5, seed=0)
        assert isinstance(result, dimod.SampleSet)

    def test_sample_bqm_returns_sampleset(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0}, {('a', 'b'): 0.5}, 0.0, 'SPIN')
        result = self.sampler.sample(bqm, num_reads=5, seed=0)
        assert isinstance(result, dimod.SampleSet)

    def test_sample_qubo_returns_sampleset(self):
        Q = {('a', 'a'): -1.0, ('b', 'b'): -1.0, ('a', 'b'): 0.5}
        result = self.sampler.sample_qubo(Q, num_reads=5, seed=0)
        assert isinstance(result, dimod.SampleSet)

    def test_num_reads_respected(self):
        for reads in [1, 10, 50]:
            result = self.sampler.sample_ising({'a': -1.0}, {}, num_reads=reads, seed=0)
            assert len(result) == reads

    def test_vartype_is_spin(self):
        result = self.sampler.sample_ising({'a': -1.0}, {}, num_reads=3, seed=0)
        assert result.vartype == dimod.SPIN

    def test_spin_values_valid(self):
        result = self.sampler.sample_ising({'a': -1.0, 'b': 1.0}, {('a', 'b'): -0.5}, num_reads=20, seed=0)
        for sample in result.samples():
            for val in sample.values():
                assert val in (-1, +1)

    def test_all_variables_present(self):
        result = self.sampler.sample_ising({'a': -1.0}, {('b', 'c'): 0.5}, num_reads=5, seed=0)
        for sample in result.samples():
            assert set(sample.keys()) == {'a', 'b', 'c'}

    def test_energy_consistent(self):
        h = {'a': -1.0, 'b': 0.5}
        J = {('a', 'b'): -0.5}
        result = self.sampler.sample_ising(h, J, num_reads=10, seed=0)
        for sample, energy in result.data(['sample', 'energy']):
            assert abs(energy - dimod.ising_energy(sample, h, J)) < 1e-8

    def test_deterministic_with_seed(self):
        h = {'a': -1.0, 'b': 1.0}
        J = {('a', 'b'): -0.5}
        r1 = self.sampler.sample_ising(h, J, num_reads=20, seed=123)
        r2 = self.sampler.sample_ising(h, J, num_reads=20, seed=123)
        assert [dict(s) for s in r1.samples()] == [dict(s) for s in r2.samples()]

    def test_empty_problem(self):
        result = self.sampler.sample_ising({}, {}, num_reads=5, seed=0)
        assert isinstance(result, dimod.SampleSet)


# ---------------------------------------------------------------------------
# Tests: sample_qubo
# ---------------------------------------------------------------------------


class TestSampleQubo:
    def setup_method(self):
        self.sampler = QuTipSampler()

    def test_returns_sampleset(self):
        Q = {('a', 'a'): -1.0, ('b', 'b'): -1.0, ('a', 'b'): 0.5}
        result = self.sampler.sample_qubo(Q, num_reads=5, seed=0)
        assert isinstance(result, dimod.SampleSet)

    def test_vartype_is_binary(self):
        Q = {('a', 'a'): -1.0, ('b', 'b'): -1.0, ('a', 'b'): 0.5}
        result = self.sampler.sample_qubo(Q, num_reads=5, seed=0)
        assert result.vartype == dimod.BINARY

    def test_binary_values_valid(self):
        Q = {('a', 'a'): -1.0, ('b', 'b'): -1.0, ('a', 'b'): 0.5}
        result = self.sampler.sample_qubo(Q, num_reads=20, seed=0)
        for sample in result.samples():
            for val in sample.values():
                assert val in (0, 1)

    def test_num_reads_respected(self):
        Q = {('x', 'x'): -1.0}
        for reads in [1, 10, 30]:
            result = self.sampler.sample_qubo(Q, num_reads=reads, seed=0)
            assert len(result) == reads

    def test_all_variables_present(self):
        Q = {('a', 'a'): -1.0, ('b', 'b'): 0.5, ('a', 'b'): 0.25}
        result = self.sampler.sample_qubo(Q, num_reads=5, seed=0)
        for sample in result.samples():
            assert set(sample.keys()) == {'a', 'b'}

    def test_energy_consistent(self):
        Q = {('a', 'a'): -1.0, ('b', 'b'): -1.0, ('a', 'b'): 0.5}
        result = self.sampler.sample_qubo(Q, num_reads=10, seed=0)
        for sample, energy in result.data(['sample', 'energy']):
            assert abs(energy - dimod.qubo_energy(sample, Q)) < 1e-8

    def test_deterministic_with_seed(self):
        Q = {('a', 'a'): -1.0, ('b', 'b'): -0.5, ('a', 'b'): 0.25}
        r1 = self.sampler.sample_qubo(Q, num_reads=20, seed=7)
        r2 = self.sampler.sample_qubo(Q, num_reads=20, seed=7)
        assert [dict(s) for s in r1.samples()] == [dict(s) for s in r2.samples()]

    def test_diagonal_only_qubo(self):
        """QUBO with only diagonal terms (no couplings)."""
        Q = {('a', 'a'): -2.0, ('b', 'b'): 1.0}
        result = self.sampler.sample_qubo(Q, num_reads=10, seed=0)
        assert result.vartype == dimod.BINARY
        assert len(result) == 10

    def test_override_anneal_params(self):
        Q = {('a', 'a'): -1.0}
        result = self.sampler.sample_qubo(Q, num_reads=5, seed=0, anneal_time=5.0, n_steps=50)
        assert isinstance(result, dimod.SampleSet)


# ---------------------------------------------------------------------------
# Tests: sample (BQM)
# ---------------------------------------------------------------------------


class TestSampleBqm:
    def setup_method(self):
        self.sampler = QuTipSampler()

    def test_returns_sampleset(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 0.5}, {('a', 'b'): -0.5}, 0.0, 'SPIN')
        result = self.sampler.sample(bqm, num_reads=5, seed=0)
        assert isinstance(result, dimod.SampleSet)

    def test_spin_bqm_preserves_vartype(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0}, {}, 0.0, 'SPIN')
        result = self.sampler.sample(bqm, num_reads=5, seed=0)
        assert result.vartype == dimod.SPIN

    def test_binary_bqm_preserves_vartype(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0}, {}, 0.0, 'BINARY')
        result = self.sampler.sample(bqm, num_reads=5, seed=0)
        assert result.vartype == dimod.BINARY

    def test_spin_values_valid(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 0.5}, {('a', 'b'): -0.5}, 0.0, 'SPIN')
        result = self.sampler.sample(bqm, num_reads=20, seed=0)
        for sample in result.samples():
            for val in sample.values():
                assert val in (-1, +1)

    def test_binary_values_valid(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 0.5}, {('a', 'b'): -0.5}, 0.0, 'BINARY')
        result = self.sampler.sample(bqm, num_reads=20, seed=0)
        for sample in result.samples():
            for val in sample.values():
                assert val in (0, 1)

    def test_num_reads_respected(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0}, {}, 0.0, 'SPIN')
        for reads in [1, 10, 30]:
            result = self.sampler.sample(bqm, num_reads=reads, seed=0)
            assert len(result) == reads

    def test_energy_consistent_spin(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 0.5}, {('a', 'b'): -0.5}, 0.0, 'SPIN')
        result = self.sampler.sample(bqm, num_reads=10, seed=0)
        for sample, energy in result.data(['sample', 'energy']):
            assert abs(energy - bqm.energy(sample)) < 1e-8

    def test_energy_consistent_binary(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 0.5}, {('a', 'b'): -0.5}, 0.0, 'BINARY')
        result = self.sampler.sample(bqm, num_reads=10, seed=0)
        for sample, energy in result.data(['sample', 'energy']):
            assert abs(energy - bqm.energy(sample)) < 1e-8

    def test_deterministic_with_seed(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -0.5}, 0.0, 'SPIN')
        r1 = self.sampler.sample(bqm, num_reads=20, seed=99)
        r2 = self.sampler.sample(bqm, num_reads=20, seed=99)
        assert [dict(s) for s in r1.samples()] == [dict(s) for s in r2.samples()]

    def test_all_variables_present(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0}, {('b', 'c'): 0.5}, 0.0, 'SPIN')
        result = self.sampler.sample(bqm, num_reads=5, seed=0)
        for sample in result.samples():
            assert set(sample.keys()) == {'a', 'b', 'c'}

    def test_override_anneal_params(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1.0}, {}, 0.0, 'SPIN')
        result = self.sampler.sample(bqm, num_reads=5, seed=0, anneal_time=5.0, n_steps=50)
        assert isinstance(result, dimod.SampleSet)

    def test_nonzero_offset_ignored_in_vartype_conversion(self):
        """BQM with offset: energies should match bqm.energy(), not raw Ising."""
        bqm = dimod.BinaryQuadraticModel({'a': -1.0}, {}, 3.0, 'SPIN')
        result = self.sampler.sample(bqm, num_reads=10, seed=0)
        for sample, energy in result.data(['sample', 'energy']):
            assert abs(energy - bqm.energy(sample)) < 1e-8


# ---------------------------------------------------------------------------
# Tests: dimod composites
# ---------------------------------------------------------------------------


class TestDimodComposites:
    """Verify QuTipSampler composes correctly with dimod's built-in composites."""

    def test_tracking_composite_records_inputs(self):
        """TrackingComposite should capture the h/J passed to the sampler."""
        sampler = dimod.TrackingComposite(QuTipSampler())
        h = {'a': -1.0, 'b': 0.5}
        J = {('a', 'b'): -0.5}
        sampler.sample_ising(h, J, num_reads=5, seed=0)
        assert sampler.input['h'] == h
        assert sampler.input['J'] == J

    def test_tracking_composite_returns_sampleset(self):
        sampler = dimod.TrackingComposite(QuTipSampler())
        result = sampler.sample_ising({'a': -1.0}, {}, num_reads=5, seed=0)
        assert isinstance(result, dimod.SampleSet)

    def test_tracking_composite_with_bqm(self):
        sampler = dimod.TrackingComposite(QuTipSampler())
        bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 0.5}, {('a', 'b'): -0.5}, 0.0, 'SPIN')
        result = sampler.sample(bqm, num_reads=5, seed=0)
        assert isinstance(result, dimod.SampleSet)
        assert result.vartype == dimod.SPIN

    def test_tracking_composite_with_qubo(self):
        sampler = dimod.TrackingComposite(QuTipSampler())
        Q = {('a', 'a'): -1.0, ('b', 'b'): -1.0, ('a', 'b'): 0.5}
        result = sampler.sample_qubo(Q, num_reads=5, seed=0)
        assert isinstance(result, dimod.SampleSet)
        assert result.vartype == dimod.BINARY

    def test_truncate_composite_limits_samples(self):
        """TruncateComposite(n=3) should return exactly 3 lowest-energy samples."""
        sampler = dimod.TruncateComposite(QuTipSampler(), n=3)
        result = sampler.sample_ising({'a': -1.0, 'b': 1.0}, {('a', 'b'): -0.5}, num_reads=20, seed=0)
        assert len(result) == 3

    def test_truncate_composite_returns_lowest_energy(self):
        """Samples from TruncateComposite must be sorted by energy ascending."""
        sampler = dimod.TruncateComposite(QuTipSampler(), n=5)
        result = sampler.sample_ising({'a': -1.0, 'b': 1.0}, {('a', 'b'): -0.5}, num_reads=30, seed=0)
        energies = [e for _, e in result.data(['sample', 'energy'])]
        assert energies == sorted(energies)

    def test_truncate_composite_with_bqm(self):
        sampler = dimod.TruncateComposite(QuTipSampler(), n=4)
        bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 0.5}, {('a', 'b'): -0.5}, 0.0, 'SPIN')
        result = sampler.sample(bqm, num_reads=20, seed=0)
        assert len(result) == 4
        assert result.vartype == dimod.SPIN

    def test_nested_composites(self):
        """TrackingComposite wrapping TruncateComposite wrapping QuTipSampler."""
        inner = dimod.TruncateComposite(QuTipSampler(), n=3)
        sampler = dimod.TrackingComposite(inner)
        result = sampler.sample_ising({'a': -1.0}, {}, num_reads=10, seed=0)
        assert isinstance(result, dimod.SampleSet)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: dwave-preprocessing composites
# ---------------------------------------------------------------------------


class TestDwavePreprocessingComposites:
    """Verify QuTipSampler composes with dwave-preprocessing composites."""

    # --- SpinReversalTransformComposite ---

    def test_srtc_returns_sampleset(self):
        sampler = SpinReversalTransformComposite(QuTipSampler())
        result = sampler.sample_ising({'a': -1.0}, {}, num_reads=5, seed=0)
        assert isinstance(result, dimod.SampleSet)

    def test_srtc_vartype_is_spin(self):
        sampler = SpinReversalTransformComposite(QuTipSampler())
        result = sampler.sample_ising({'a': -1.0, 'b': 1.0}, {('a', 'b'): -0.5}, num_reads=5, seed=0)
        assert result.vartype == dimod.SPIN

    def test_srtc_spin_values_valid(self):
        sampler = SpinReversalTransformComposite(QuTipSampler())
        result = sampler.sample_ising({'a': -1.0, 'b': 1.0}, {('a', 'b'): -0.5}, num_reads=20, seed=0)
        for sample in result.samples():
            for val in sample.values():
                assert val in (-1, +1)

    def test_srtc_with_bqm_spin(self):
        sampler = SpinReversalTransformComposite(QuTipSampler())
        bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 0.5}, {('a', 'b'): -0.5}, 0.0, 'SPIN')
        result = sampler.sample(bqm, num_reads=5, seed=0)
        assert result.vartype == dimod.SPIN

    def test_srtc_with_qubo(self):
        sampler = SpinReversalTransformComposite(QuTipSampler())
        Q = {('a', 'a'): -1.0, ('b', 'b'): -1.0, ('a', 'b'): 0.5}
        result = sampler.sample_qubo(Q, num_reads=5, seed=0)
        assert result.vartype == dimod.BINARY

    def test_srtc_num_transforms_multiplies_reads(self):
        """num_spin_reversal_transforms=k runs k independent transforms, each
        with num_reads samples, so the total should be k * num_reads."""
        sampler = SpinReversalTransformComposite(QuTipSampler())
        result = sampler.sample_ising(
            {'a': -1.0}, {}, num_reads=5, seed=0, num_spin_reversal_transforms=3
        )
        assert len(result) == 15

    def test_srtc_energy_consistent(self):
        h = {'a': -1.0, 'b': 0.5}
        J = {('a', 'b'): -0.5}
        sampler = SpinReversalTransformComposite(QuTipSampler())
        result = sampler.sample_ising(h, J, num_reads=10, seed=0)
        for sample, energy in result.data(['sample', 'energy']):
            assert abs(energy - dimod.ising_energy(sample, h, J)) < 1e-8

    # --- ScaleComposite ---

    def test_scale_returns_sampleset(self):
        sampler = ScaleComposite(QuTipSampler())
        result = sampler.sample_ising({'a': -1.0}, {}, num_reads=5, seed=0, scalar=0.5)
        assert isinstance(result, dimod.SampleSet)

    def test_scale_vartype_is_spin(self):
        sampler = ScaleComposite(QuTipSampler())
        result = sampler.sample_ising({'a': -1.0}, {}, num_reads=5, seed=0, scalar=0.5)
        assert result.vartype == dimod.SPIN

    def test_scale_energy_consistent(self):
        """Energies in the result must match the original (unscaled) problem."""
        h = {'a': -1.0, 'b': 0.5}
        J = {('a', 'b'): -0.5}
        sampler = ScaleComposite(QuTipSampler())
        result = sampler.sample_ising(h, J, num_reads=10, seed=0, scalar=0.5)
        for sample, energy in result.data(['sample', 'energy']):
            assert abs(energy - dimod.ising_energy(sample, h, J)) < 1e-8

    def test_scale_with_bqm(self):
        sampler = ScaleComposite(QuTipSampler())
        bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 0.5}, {('a', 'b'): -0.5}, 0.0, 'SPIN')
        result = sampler.sample(bqm, num_reads=5, seed=0, scalar=0.5)
        assert result.vartype == dimod.SPIN

    # --- Composed together ---

    def test_srtc_wrapping_scale(self):
        """SpinReversalTransformComposite(ScaleComposite(QuTipSampler()))."""
        sampler = SpinReversalTransformComposite(ScaleComposite(QuTipSampler()))
        result = sampler.sample_ising(
            {'a': -1.0, 'b': 1.0}, {('a', 'b'): -0.5},
            num_reads=5, seed=0, scalar=0.5,
        )
        assert isinstance(result, dimod.SampleSet)
        assert result.vartype == dimod.SPIN


# ---------------------------------------------------------------------------
# Statistical sanity tests
# ---------------------------------------------------------------------------

class TestStatisticalSanity:
    @pytest.mark.slow
    def test_single_qubit_ground_state(self):
        """h={'a': -1.0} -> ground state is s_a=+1. Should dominate samples."""
        sampler = QuTipSampler(anneal_time=20.0, n_steps=400)
        result = sampler.sample_ising({'a': -1.0}, {}, num_reads=500, seed=0)
        spins = [s['a'] for s in result.samples()]
        assert spins.count(+1) / len(spins) > 0.6

    @pytest.mark.slow
    def test_two_qubit_antiferromagnet(self):
        """J={(a,b): +1.0} -> anti-aligned states should dominate."""
        sampler = QuTipSampler(anneal_time=20.0, n_steps=400)
        result = sampler.sample_ising({}, {('a', 'b'): 1.0}, num_reads=500, seed=42)
        aligned = sum(1 for s in result.samples() if s['a'] == s['b'])
        assert aligned / len(result) < 0.5
