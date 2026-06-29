import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from orbital_state import OrbitalState

DIM = 8


def rand_vec(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM)
    return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------

class TestInitialize:

    def test_position_set_to_embedding(self):
        emb = rand_vec(0)
        state = OrbitalState.initialize(emb)
        assert np.allclose(state.position, emb)

    def test_velocity_is_zero(self):
        state = OrbitalState.initialize(rand_vec(1))
        assert np.allclose(state.velocity, np.zeros(DIM))

    def test_acceleration_is_zero(self):
        state = OrbitalState.initialize(rand_vec(2))
        assert np.allclose(state.acceleration, np.zeros(DIM))

    def test_does_not_alias_embedding(self):
        emb = rand_vec(3)
        state = OrbitalState.initialize(emb)
        emb[0] = 9999.0
        assert state.position[0] != 9999.0


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:

    def test_position_updated_to_new_position(self):
        emb0 = rand_vec(10)
        state = OrbitalState.initialize(emb0)
        emb1 = rand_vec(11)
        state.update(emb1)
        assert np.allclose(state.position, emb1)

    def test_velocity_equals_new_minus_old_position(self):
        emb0 = rand_vec(20)
        state = OrbitalState.initialize(emb0)
        emb1 = rand_vec(21)
        state.update(emb1)
        expected_vel = emb1 - emb0
        assert np.allclose(state.velocity, expected_vel)

    def test_acceleration_equals_new_minus_old_velocity(self):
        emb0 = rand_vec(30)
        state = OrbitalState.initialize(emb0)
        emb1 = rand_vec(31)
        state.update(emb1)
        vel_after_first = emb1 - emb0
        emb2 = rand_vec(32)
        state.update(emb2)
        expected_acc = (emb2 - emb1) - vel_after_first
        assert np.allclose(state.acceleration, expected_acc)

    def test_does_not_alias_new_position(self):
        state = OrbitalState.initialize(rand_vec(40))
        emb1 = rand_vec(41)
        state.update(emb1)
        emb1[0] = 9999.0
        assert state.position[0] != 9999.0

    def test_three_identical_updates_give_zero_acceleration(self):
        # After the first two identical updates: velocity=0. Third identical update:
        # new_velocity = 0, acceleration = 0 - 0 = 0.
        state = OrbitalState.initialize(rand_vec(50))
        emb1 = rand_vec(51)
        state.update(emb1)
        state.update(emb1)  # velocity -> 0
        state.update(emb1)  # acceleration = new_vel - old_vel = 0 - 0 = 0
        assert np.allclose(state.acceleration, np.zeros(DIM), atol=1e-10)


# ---------------------------------------------------------------------------
# momentum property
# ---------------------------------------------------------------------------

class TestMomentum:

    def test_momentum_is_norm_of_velocity(self):
        state = OrbitalState.initialize(rand_vec(60))
        emb1 = rand_vec(61)
        state.update(emb1)
        expected = float(np.linalg.norm(state.velocity))
        assert state.momentum == pytest.approx(expected)

    def test_momentum_zero_after_initialize(self):
        state = OrbitalState.initialize(rand_vec(62))
        assert state.momentum == pytest.approx(0.0)

    def test_momentum_positive_after_move(self):
        state = OrbitalState.initialize(rand_vec(63))
        state.update(rand_vec(64))
        assert state.momentum > 0.0


# ---------------------------------------------------------------------------
# speed_of_change property
# ---------------------------------------------------------------------------

class TestSpeedOfChange:

    def test_speed_of_change_is_norm_of_acceleration(self):
        state = OrbitalState.initialize(rand_vec(70))
        state.update(rand_vec(71))
        state.update(rand_vec(72))
        expected = float(np.linalg.norm(state.acceleration))
        assert state.speed_of_change == pytest.approx(expected)

    def test_speed_of_change_zero_after_initialize(self):
        state = OrbitalState.initialize(rand_vec(73))
        assert state.speed_of_change == pytest.approx(0.0)

    def test_nonzero_acceleration_after_direction_change(self):
        state = OrbitalState.initialize(np.zeros(DIM))
        # First update: move in positive direction
        v1 = np.zeros(DIM); v1[0] = 1.0
        state.update(v1)
        # Second update: move in orthogonal direction — acceleration must be nonzero
        v2 = np.zeros(DIM); v2[1] = 1.0
        state.update(v2)
        assert state.speed_of_change > 0.0
