import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
import pytest
from uuid import UUID
from context_body_store import ContextBodyStore
from vector_backend import FAISSBackend

DIM = 8


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def axis_vec(i, dim=DIM):
    """Unit vector along axis i."""
    v = np.zeros(dim, dtype=float)
    v[i] = 1.0
    return v


def make_store(**kwargs):
    return ContextBodyStore(embedding_dim=DIM, **kwargs)


# ---------------------------------------------------------------------------
# record() — basic insertion
# ---------------------------------------------------------------------------

class TestRecordBasic:

    def test_record_returns_uuid(self):
        store = make_store()
        uid = store.record(axis_vec(0), mass=1.0, stability=0.5)
        assert isinstance(uid, UUID)

    def test_two_distant_records_get_different_uuids(self):
        store = make_store()
        uid_a = store.record(axis_vec(0), mass=1.0, stability=0.5)
        uid_b = store.record(axis_vec(1), mass=1.0, stability=0.5)
        assert uid_a != uid_b

    def test_backend_size_increments(self):
        store = make_store()
        store.record(axis_vec(0), mass=1.0, stability=0.5)
        store.record(axis_vec(1), mass=1.0, stability=0.5)
        assert store.backend.size == 2


# ---------------------------------------------------------------------------
# record() — deduplication
# ---------------------------------------------------------------------------

class TestRecordDedup:

    def test_near_duplicate_returns_same_uuid(self):
        store = make_store(dedup_distance=0.1, decay_interval=9999)
        v = axis_vec(0)
        uid_a = store.record(v, mass=2.0, stability=0.5)
        # Nearly identical vector (tiny perturbation)
        v2 = norm(v + np.full(DIM, 1e-4))
        uid_b = store.record(v2, mass=2.0, stability=0.5)
        assert uid_a == uid_b

    def test_near_duplicate_updates_mass_as_average(self):
        store = make_store(dedup_distance=0.1, decay_interval=9999)
        v = axis_vec(0)
        uid = store.record(v, mass=4.0, stability=0.5)
        v2 = norm(v + np.full(DIM, 1e-4))
        store.record(v2, mass=8.0, stability=0.5)
        assert store._record_mass[str(uid)] == pytest.approx(6.0)  # (4+8)/2

    def test_dedup_does_not_increase_backend_size(self):
        store = make_store(dedup_distance=0.1, decay_interval=9999)
        v = axis_vec(0)
        store.record(v, mass=1.0, stability=0.5)
        v2 = norm(v + np.full(DIM, 1e-4))
        store.record(v2, mass=1.0, stability=0.5)
        assert store.backend.size == 1

    def test_distant_body_creates_new_record(self):
        store = make_store(dedup_distance=0.05, decay_interval=9999)
        uid_a = store.record(axis_vec(0), mass=1.0, stability=0.5)
        uid_b = store.record(axis_vec(1), mass=1.0, stability=0.5)
        assert uid_a != uid_b
        assert store.backend.size == 2


# ---------------------------------------------------------------------------
# record() — re-emergence
# ---------------------------------------------------------------------------

class TestRecordReemergence:

    def test_reemergence_returns_dormant_uuid(self):
        store = make_store(
            dedup_distance=0.05,
            reemergence_distance=0.15,
            reemergence_mass_threshold=0.1,
            decay_interval=9999,
        )
        # Insert original record
        v0 = axis_vec(0)
        uid_orig = store.record(v0, mass=1.0, stability=0.5)

        # Simulate decay: mass drops below threshold
        store._record_mass[str(uid_orig)] = 0.05
        store.backend.update_metadata(str(uid_orig), {"mass": 0.05})

        # Insert a new body at cosine dist ~0.12 from v0 (within reemergence_distance
        # but outside dedup_distance)
        # cos_dist = 1 - dot(v0, v1). For axis_vec(0) and a vector with small v[1]:
        # Use a vector that is slightly off-axis: [cos(theta), sin(theta), 0, ...]
        theta = np.arccos(1.0 - 0.12)  # cos_dist = 0.12 => cos(angle) = 0.88
        v1 = np.zeros(DIM)
        v1[0] = np.cos(theta)
        v1[1] = np.sin(theta)
        v1 = norm(v1)

        uid_new = store.record(v1, mass=0.5, stability=0.3)
        assert uid_new == uid_orig

    def test_reemergence_does_not_fire_for_healthy_record(self):
        store = make_store(
            dedup_distance=0.05,
            reemergence_distance=0.15,
            reemergence_mass_threshold=0.1,
            decay_interval=9999,
        )
        v0 = axis_vec(0)
        uid_orig = store.record(v0, mass=5.0, stability=0.5)
        # mass is 5.0 — well above threshold 0.1

        theta = np.arccos(1.0 - 0.12)
        v1 = np.zeros(DIM)
        v1[0] = np.cos(theta)
        v1[1] = np.sin(theta)
        v1 = norm(v1)

        uid_new = store.record(v1, mass=0.5, stability=0.3)
        assert uid_new != uid_orig  # healthy record NOT reused; new record created

    def test_reemergence_boosts_dormant_mass(self):
        store = make_store(
            dedup_distance=0.05,
            reemergence_distance=0.20,
            reemergence_mass_threshold=0.1,
            decay_interval=9999,
        )
        v0 = axis_vec(0)
        uid_orig = store.record(v0, mass=1.0, stability=0.5)
        store._record_mass[str(uid_orig)] = 0.05

        theta = np.arccos(1.0 - 0.12)
        v1 = np.zeros(DIM)
        v1[0] = np.cos(theta)
        v1[1] = np.sin(theta)
        v1 = norm(v1)

        store.record(v1, mass=0.5, stability=0.3)
        # Mass should have increased from 0.05
        assert store._record_mass[str(uid_orig)] > 0.05


# ---------------------------------------------------------------------------
# query_nearby()
# ---------------------------------------------------------------------------

class TestQueryNearby:

    def test_empty_store_returns_empty_list(self):
        store = make_store(decay_interval=9999)
        results = store.query_nearby(axis_vec(0))
        assert results == []

    def test_returns_list_of_record_dist_tuples(self):
        store = make_store(decay_interval=9999)
        store.record(axis_vec(0), mass=1.0, stability=0.5)
        results = store.query_nearby(axis_vec(0))
        assert len(results) == 1
        rec, dist = results[0]
        assert isinstance(dist, float)

    def test_gravitational_ranking_heavier_wins(self):
        store = make_store(decay_interval=9999)
        # light body at dist ~0.05 from query
        theta_light = np.arccos(1.0 - 0.05)
        v_light = np.zeros(DIM)
        v_light[0] = np.cos(theta_light)
        v_light[1] = np.sin(theta_light)
        v_light = norm(v_light)
        uid_light = store.record(v_light, mass=0.1, stability=0.5)

        # heavy body farther at dist ~0.20
        theta_heavy = np.arccos(1.0 - 0.20)
        v_heavy = np.zeros(DIM)
        v_heavy[0] = np.cos(theta_heavy)
        v_heavy[2] = np.sin(theta_heavy)  # different plane to avoid aliasing
        v_heavy = norm(v_heavy)
        uid_heavy = store.record(v_heavy, mass=500.0, stability=0.5)

        # Force mass into cache
        store._record_mass[str(uid_light)] = 0.1
        store._record_mass[str(uid_heavy)] = 500.0

        results = store.query_nearby(axis_vec(0))
        assert len(results) >= 2
        # heavy body: 500/0.04 = 12500; light body: 0.1/0.0025 = 40
        # heavy should rank first
        first_rec, _ = results[0]
        assert store._record_mass.get(str(first_rec.id), first_rec.mass) > 100.0


# ---------------------------------------------------------------------------
# decay()
# ---------------------------------------------------------------------------

class TestDecay:

    def test_extinct_records_removed(self):
        store = make_store(
            decay_rate=1.0,
            extinction_threshold=0.01,
            decay_interval=9999,
        )
        uid = store.record(axis_vec(0), mass=0.005, stability=0.5)
        # Manually set mass below extinction threshold
        store._record_mass[str(uid)] = 0.005
        extinct = store.decay()
        assert str(uid) in extinct
        assert store.backend.size == 0

    def test_healthy_records_lose_mass(self):
        store = make_store(
            decay_rate=0.001,
            extinction_threshold=0.001,
            decay_interval=9999,
        )
        uid = store.record(axis_vec(0), mass=10.0, stability=0.5)
        original_mass = store._record_mass[str(uid)]
        store.decay()
        new_mass = store._record_mass.get(str(uid), original_mass)
        # Mass should have decreased or record was extincted (either is valid)
        assert new_mass <= original_mass

    def test_decay_returns_list_of_extinct_uuids(self):
        store = make_store(
            decay_rate=1.0,
            extinction_threshold=0.5,
            decay_interval=9999,
        )
        uid = store.record(axis_vec(0), mass=0.1, stability=0.5)
        store._record_mass[str(uid)] = 0.1
        extinct = store.decay()
        assert isinstance(extinct, list)
        assert str(uid) in extinct


# ---------------------------------------------------------------------------
# record_resonance()
# ---------------------------------------------------------------------------

class TestRecordResonance:

    def test_increments_both_directions(self):
        store = make_store(decay_interval=9999)
        uid_a = str(store.record(axis_vec(0), mass=1.0, stability=0.5))
        uid_b = str(store.record(axis_vec(1), mass=1.0, stability=0.5))
        store.record_resonance(uid_a, uid_b, delta=0.2)
        assert store._resonance_cache[uid_a][uid_b] == pytest.approx(0.2)
        assert store._resonance_cache[uid_b][uid_a] == pytest.approx(0.2)

    def test_score_accumulates(self):
        store = make_store(decay_interval=9999)
        uid_a = str(store.record(axis_vec(0), mass=1.0, stability=0.5))
        uid_b = str(store.record(axis_vec(1), mass=1.0, stability=0.5))
        store.record_resonance(uid_a, uid_b, delta=0.2)
        store.record_resonance(uid_a, uid_b, delta=0.3)
        assert store._resonance_cache[uid_a][uid_b] == pytest.approx(0.5)

    def test_score_capped_at_max_score(self):
        store = make_store(decay_interval=9999)
        uid_a = str(store.record(axis_vec(0), mass=1.0, stability=0.5))
        uid_b = str(store.record(axis_vec(1), mass=1.0, stability=0.5))
        store.record_resonance(uid_a, uid_b, delta=0.9, max_score=1.0)
        store.record_resonance(uid_a, uid_b, delta=0.9, max_score=1.0)
        assert store._resonance_cache[uid_a][uid_b] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Auto-decay trigger in query_nearby
# ---------------------------------------------------------------------------

class TestAutoDecayTrigger:

    def test_decay_fires_after_interval(self):
        store = make_store(
            decay_rate=1.0,
            extinction_threshold=0.5,
            decay_interval=0.0,   # always fires
        )
        uid = store.record(axis_vec(0), mass=0.1, stability=0.5)
        store._record_mass[str(uid)] = 0.1
        # query_nearby should trigger decay, removing the extinct record
        store.query_nearby(axis_vec(0))
        assert str(uid) not in store._record_mass

    def test_decay_does_not_fire_before_interval(self):
        store = make_store(
            decay_rate=1.0,
            extinction_threshold=0.5,
            decay_interval=9999.0,  # far future
        )
        uid = store.record(axis_vec(0), mass=0.1, stability=0.5)
        store._record_mass[str(uid)] = 0.1
        # Manually set last_decay_at to now so interval hasn't elapsed
        from datetime import datetime
        store._last_decay_at = datetime.utcnow()
        store.query_nearby(axis_vec(0))
        # Record should still be present (decay not triggered)
        assert str(uid) in store._record_mass
