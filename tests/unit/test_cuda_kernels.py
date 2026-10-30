"""Unit tests for the CUDA kernels in rl/cuda (Numba).

Without an NVIDIA GPU the kernels run in Numba's CUDA simulator (rl/cuda/backend.py sets it up), so these
tests use small instances. Every kernel is compared with the CPU implementation it mirrors.
"""
from __future__ import annotations

import unittest

import numpy as np

try:
    from rl.cuda import backend, dp as cuda_dp, pg as cuda_pg
except ImportError as e:  # numba missing
    backend = cuda_dp = cuda_pg = None
    IMPORT_ERROR = e

from rl.dp import solvers
from rl.dp.shuttle import ShuttleParams, build_mdp
from rl.pg.envs import preprocess
from rl.pg.reinforce import discounted_returns
from tests.unit.test_pg_policy import FakePong

# Small on purpose: in the simulator every CUDA thread is a Python thread (the whole module takes ~30 s there).
SMALL = ShuttleParams(max_customers=16, gamma=0.6)
EVAL_TOL = 1e-10 if backend is not None and backend.SIMULATOR else 1e-13


@unittest.skipIf(cuda_dp is None, "numba is not installed")
class LaunchHelperTests(unittest.TestCase):
    def test_grid_initialisation(self):
        self.assertEqual(backend.launch_1d(1000, 256), (4, 256))
        self.assertEqual(backend.launch_1d(1, 256), (1, 256))
        self.assertEqual(backend.launch_2d(201, 2, (32, 4)), ((7, 1), (32, 4)))
        self.assertEqual(backend.launch_3d(80, 80, 3, (16, 16, 1)), ((5, 5, 3), (16, 16, 1)))
        self.assertEqual(backend.block_for_states(201), 256)
        self.assertEqual(backend.block_for_states(5), 32)
        self.assertEqual(backend.block_for_states(1024), 1024)
        self.assertIsNone(backend.block_for_states(1025))
        self.assertIn(backend.backend_name(), ("cuda", "cuda-simulator"))


@unittest.skipIf(cuda_dp is None, "numba is not installed")
class ShuttleTableTests(unittest.TestCase):
    def test_device_built_tables_match_build_mdp(self):
        for params in (SMALL, SMALL.with_(holding_charged="end_of_period", max_dispatch=2), SMALL.with_(arrivals=(0, 2, 2, 7), max_customers=12)):
            cost, idx, p = cuda_dp.shuttle_tables_gpu(params)
            reference = build_mdp(params)
            rebuilt = cuda_dp.mdp_from_tables(cost, idx, p, params.gamma, reference.action_labels)
            np.testing.assert_allclose(rebuilt.costs, reference.costs, atol=1e-12)
            np.testing.assert_allclose(rebuilt.transitions, reference.transitions, atol=1e-12)

    def test_sparse_tables_round_trip(self):
        mdp = build_mdp(SMALL)
        cost, idx, p = cuda_dp.sparse_tables(mdp)
        self.assertEqual(idx.shape, (17, 2, 5))
        rebuilt = cuda_dp.mdp_from_tables(cost, idx, p, mdp.gamma, mdp.action_labels)
        np.testing.assert_allclose(rebuilt.transitions, mdp.transitions)


@unittest.skipIf(cuda_dp is None, "numba is not installed")
class SolverKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mdp = build_mdp(SMALL)
        cls.cpu_vi = solvers.value_iteration(cls.mdp, tol=1e-9)
        cls.cpu_pi = solvers.policy_iteration(cls.mdp)

    def test_persistent_value_iteration_matches_the_cpu_solver(self):
        gpu = cuda_dp.value_iteration_gpu(self.mdp, tol=1e-9)
        self.assertTrue(gpu.info["converged"])
        self.assertIn("persistent", gpu.info["kernel"])
        np.testing.assert_allclose(gpu.values, self.cpu_vi.values, atol=1e-8)
        np.testing.assert_array_equal(gpu.policy, self.cpu_vi.policy)
        self.assertLessEqual(abs(gpu.iterations - self.cpu_vi.iterations), 1)
        self.assertLess(gpu.info["bellman_residual"], 1e-8)
        np.testing.assert_allclose(gpu.history["‖V_{n+1} − V_n‖∞"][:5], self.cpu_vi.history["‖V_{n+1} − V_n‖∞"][:5], rtol=1e-9)

    def test_grid_path_matches_the_persistent_kernel(self):
        grid = cuda_dp.value_iteration_gpu(self.mdp, tol=1e-9, force_grid=True)
        self.assertIn("grid", grid.info["kernel"])
        np.testing.assert_allclose(grid.values, self.cpu_vi.values, atol=1e-8)
        np.testing.assert_array_equal(grid.policy, self.cpu_vi.policy)

    def test_enumeration_matches(self):
        for horizon in (0, 1, 5):
            cpu = solvers.enumeration(self.mdp, horizon)
            gpu = cuda_dp.enumeration_gpu(self.mdp, horizon)
            np.testing.assert_allclose(gpu.values, cpu.values, atol=1e-10, err_msg=f"horizon {horizon}")
            self.assertEqual(gpu.iterations, horizon)
            if horizon:
                np.testing.assert_array_equal(gpu.policy, cpu.policy)
                np.testing.assert_allclose(gpu.history["max |V_t − V_{t+1}|"], cpu.history["max |V_t − V_{t+1}|"], rtol=1e-9)
        grid = cuda_dp.enumeration_gpu(self.mdp, 3, force_grid=True)
        np.testing.assert_allclose(grid.values, solvers.enumeration(self.mdp, 3).values, atol=1e-10)

    def test_policy_evaluation_and_policy_iteration_match(self):
        V = cuda_dp.evaluate_policy_gpu(self.mdp, self.cpu_pi.policy, tol=EVAL_TOL)
        np.testing.assert_allclose(V, self.cpu_pi.values, atol=1e-8)
        gpu = cuda_dp.policy_iteration_gpu(self.mdp, eval_tol=EVAL_TOL)
        self.assertTrue(gpu.info["converged"])
        np.testing.assert_array_equal(gpu.policy, self.cpu_pi.policy)
        np.testing.assert_allclose(gpu.values, self.cpu_pi.values, atol=1e-8)
        self.assertEqual(gpu.info["threshold"], self.cpu_pi.info["threshold"])
        self.assertEqual(gpu.history["states whose action changed"][-1], 0.0)

    def test_simulation_kernel_reproduces_the_value(self):
        sim = cuda_dp.simulate_policy_gpu(self.mdp, self.cpu_pi.policy, start=5, horizon=50, episodes=96, seed=3, threads=32)
        self.assertEqual(sim["costs"].shape, (96,))
        self.assertLess(abs(sim["mean"] - self.cpu_pi.values[5]), 4 * sim["stderr"] + sim["truncation_bound"])
        again = cuda_dp.simulate_policy_gpu(self.mdp, self.cpu_pi.policy, start=5, horizon=50, episodes=96, seed=3, threads=32)
        np.testing.assert_array_equal(sim["costs"], again["costs"])  # reproducible per-thread streams
        self.assertEqual(sim["grid"], {"blocks": 3, "threads": 32})

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            cuda_dp.value_iteration_gpu(build_mdp(SMALL.with_(gamma=1.0)))


@unittest.skipIf(cuda_pg is None, "numba is not installed")
class ReturnsScanTests(unittest.TestCase):
    def test_single_chunk_and_multi_chunk_scans_match_the_recursion(self):
        rng = np.random.default_rng(0)
        for T, chunk in ((1, 64), (37, 64), (64, 64), (65, 64), (130, 32)):
            r = rng.normal(size=T)
            for gamma in (0.0, 0.95, 1.0):
                gpu = cuda_pg.discounted_returns_gpu(r, gamma, chunk=chunk)
                np.testing.assert_allclose(gpu, discounted_returns(r, gamma), rtol=1e-12, atol=1e-12, err_msg=f"T={T}, chunk={chunk}, gamma={gamma}")

    def test_edge_cases(self):
        self.assertEqual(cuda_pg.discounted_returns_gpu([], 0.9).shape, (0,))
        with self.assertRaises(ValueError):
            cuda_pg.discounted_returns_gpu([1.0], 0.9, chunk=0)


@unittest.skipIf(cuda_pg is None, "numba is not installed")
class PreprocessKernelTests(unittest.TestCase):
    def test_matches_the_handout_preprocess_and_frame_difference(self):
        fake = FakePong(steps=3)
        frames = [fake.frame()]
        for _ in range(2):
            fake.t += 1
            frames.append(fake.frame())
        frames = np.stack(frames)
        expected_bin = np.stack([preprocess(f.copy()) for f in frames])
        binarized, diff = cuda_pg.preprocess_frames_gpu(frames)
        np.testing.assert_array_equal(binarized, expected_bin)
        np.testing.assert_array_equal(diff, 0.0)
        binarized2, diff2 = cuda_pg.preprocess_frames_gpu(frames[1:], previous=expected_bin[:-1])
        np.testing.assert_array_equal(binarized2, expected_bin[1:])
        np.testing.assert_array_equal(diff2, expected_bin[1:] - expected_bin[:-1])
        self.assertGreater(np.abs(diff2).sum(), 0)
        single_bin, single_diff = cuda_pg.preprocess_frames_gpu(frames[0])
        self.assertEqual(single_bin.shape, (1, 80, 80))


@unittest.skipIf(cuda_pg is None, "numba is not installed")
class AdapterKernelPathTests(unittest.TestCase):
    def test_pong_adapter_with_kernels_matches_the_numpy_adapter(self):
        from rl.pg.envs import make_adapter
        from rl.pg.reinforce import TrainConfig

        plain_env, kernel_env = FakePong(steps=3), FakePong(steps=3)
        plain = make_adapter(TrainConfig(env="pong", part=1, gamma=0.99), plain_env)
        fast = make_adapter(TrainConfig(env="pong", part=1, gamma=0.99, extra={"kernels": True}), kernel_env)
        np.testing.assert_array_equal(fast.reset(kernel_env), plain.reset(plain_env))
        for action in (0, 1, 0):
            obs_plain = plain.step(plain_env, action)[0]
            obs_fast = fast.step(kernel_env, action)[0]
            np.testing.assert_array_equal(obs_fast, obs_plain)
        self.assertGreater(np.abs(obs_fast).sum(), 0)


@unittest.skipIf(cuda_pg is None, "numba is not installed")
class ReductionKernelTests(unittest.TestCase):
    def test_mean_std_and_normalisation(self):
        rng = np.random.default_rng(1)
        x = rng.normal(loc=50.0, scale=3.0, size=300)
        mean, std = cuda_pg.mean_std_gpu(x, threads=64)
        self.assertAlmostEqual(mean, x.mean(), places=9)
        self.assertAlmostEqual(std, x.std(), places=9)
        z = cuda_pg.normalize_gpu(x)
        np.testing.assert_allclose(z, (x - x.mean()) / x.std(), rtol=1e-9, atol=1e-9)
        constant = cuda_pg.normalize_gpu(np.full(10, 3.0))
        np.testing.assert_allclose(constant, 0.0, atol=1e-12)  # constant input: centred, not divided by ~0
        self.assertEqual(cuda_pg.normalize_gpu([]).shape, (0,))

    def test_softmax_sampling(self):
        logits = np.array([[0.0, 0.0], [5.0, 0.0], [-3.0, 2.0, 1.0 - 1e9]][:2])
        actions, probs = cuda_pg.sample_actions_gpu(logits, [0.3, 0.999])
        np.testing.assert_allclose(probs[0], [0.5, 0.5])
        np.testing.assert_allclose(probs[1], np.exp([5.0, 0.0]) / np.exp([5.0, 0.0]).sum())
        self.assertEqual(actions.tolist(), [0, 1])
        rng = np.random.default_rng(0)
        many = np.tile([[0.0, 1.0, 2.0]], (800, 1))
        acts, _ = cuda_pg.sample_actions_gpu(many, rng.random(800))
        freq = np.bincount(acts, minlength=3) / 800
        np.testing.assert_allclose(freq, np.exp([0, 1, 2]) / np.exp([0, 1, 2]).sum(), atol=0.06)
        with self.assertRaises(ValueError):
            cuda_pg.sample_actions_gpu(many, rng.random(5))


if __name__ == "__main__":
    unittest.main()
