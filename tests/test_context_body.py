import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from context_body import ContextBody

DIM = 8


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def make_embedding(val=1.0, dim=DIM):
    v = np.zeros(dim)
    v[0] = val
    return norm(v)


# ---------------------------------------------------------------------------
# accrete() — mass accumulation
# ---------------------------------------------------------------------------

class TestAccreteMass:

    def test_first_accrete_sets_mass(self):
        body = ContextBody()
        emb = make_embedding()
        body.accrete(token_id=0, token_embedding=emb, token_mass=3.0)
        assert body.mass == pytest.approx(3.0)

    def test_mass_accumulates_across_calls(self):
        body = ContextBody()
        emb = make_embedding()
        body.accrete(0, emb, token_mass=2.0)
        body.accrete(1, emb, token_mass=5.0)
        assert body.mass == pytest.approx(7.0)

    def test_default_token_mass_is_one(self):
        body = ContextBody()
        emb = make_embedding()
        body.accrete(token_id=0, token_embedding=emb)
        assert body.mass == pytest.approx(1.0)

    def test_multiple_calls_sum_all_masses(self):
        body = ContextBody()
        masses = [1.5, 2.5, 3.0, 0.5]
        for i, m in enumerate(masses):
            body.accrete(i, make_embedding(float(i + 1)), token_mass=m)
        assert body.mass == pytest.approx(sum(masses))


# ---------------------------------------------------------------------------
# accrete() — centroid updates
# ---------------------------------------------------------------------------

class TestAccreteCentroid:

    def test_first_accrete_sets_centroid(self):
        body = ContextBody()
        emb = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        body.accrete(0, emb)
        assert np.allclose(body.centroid, emb, atol=1e-6)

    def test_centroid_is_running_mean(self):
        body = ContextBody()
        e0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        e1 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        body.accrete(0, e0)
        body.accrete(1, e1)
        expected = (e0 + e1) / 2
        assert np.allclose(body.centroid, expected, atol=1e-6)

    def test_centroid_tracks_mean_of_all_tokens(self):
        body = ContextBody()
        rng = np.random.default_rng(42)
        embs = rng.standard_normal((5, DIM))
        for i, e in enumerate(embs):
            body.accrete(i, e)
        assert np.allclose(body.centroid, embs.mean(axis=0), atol=1e-6)


# ---------------------------------------------------------------------------
# accrete() — orbital_radii
# ---------------------------------------------------------------------------

class TestAccreteOrbitalRadii:

    def test_orbital_radius_recorded_per_token(self):
        body = ContextBody()
        emb = make_embedding()
        body.accrete(token_id=7, token_embedding=emb)
        assert 7 in body.orbital_radii

    def test_orbital_radius_is_float(self):
        body = ContextBody()
        body.accrete(0, make_embedding())
        assert isinstance(body.orbital_radii[0], float)

    def test_orbital_radii_for_multiple_tokens(self):
        body = ContextBody()
        for i in range(4):
            e = norm(np.eye(DIM)[i % DIM])
            body.accrete(i, e)
        assert set(body.orbital_radii.keys()) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# accrete() — centroid_velocity
# ---------------------------------------------------------------------------

class TestAccreteCentroidVelocity:

    def test_velocity_updated_after_second_accrete(self):
        body = ContextBody()
        e0 = norm(np.array([1.0, 0.0] + [0.0] * (DIM - 2)))
        e1 = norm(np.array([0.0, 1.0] + [0.0] * (DIM - 2)))
        body.accrete(0, e0)
        body.accrete(1, e1)
        # velocity = new_centroid - prev_centroid; should be non-zero when direction changes
        assert not np.allclose(body.centroid_velocity, np.zeros(DIM), atol=1e-8)

    def test_velocity_is_zero_when_centroid_unchanged(self):
        body = ContextBody()
        e = norm(np.ones(DIM))
        body.accrete(0, e)
        body.accrete(1, e)  # same direction: centroid barely moves
        # velocity should be very small (centroid barely moves when adding same vector)
        assert np.linalg.norm(body.centroid_velocity) < 0.2


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------

class TestClassify:

    @pytest.mark.parametrize("mass,expected", [
        (0.0,    "asteroid"),
        (5.0,    "asteroid"),
        (9.9,    "asteroid"),
        (10.1,   "moon"),
        (50.0,   "moon"),
        (100.1,  "planet"),
        (300.0,  "planet"),
        (500.1,  "neutron_star"),
        (999.9,  "neutron_star"),
        (1000.1, "black_hole"),
        (9999.0, "black_hole"),
    ])
    def test_classify_tiers(self, mass, expected):
        body = ContextBody()
        body.mass = mass
        assert body.classify() == expected

    def test_classify_boundary_10_is_moon(self):
        body = ContextBody()
        body.mass = 10.1
        assert body.classify() == "moon"

    def test_classify_boundary_100_is_planet(self):
        body = ContextBody()
        body.mass = 100.1
        assert body.classify() == "planet"

    def test_classify_boundary_500_is_neutron_star(self):
        body = ContextBody()
        body.mass = 500.1
        assert body.classify() == "neutron_star"

    def test_classify_boundary_1000_is_black_hole(self):
        body = ContextBody()
        body.mass = 1000.1
        assert body.classify() == "black_hole"
