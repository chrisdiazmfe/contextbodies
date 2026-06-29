import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

# torch is optional for tests; skip entire module if unavailable
torch = pytest.importorskip("torch")

from datetime import datetime, timedelta
from uuid import uuid4

from context_body import ContextBody
from context_body_record import ContextBodyRecord
from context_body_store import ContextBodyStore
from gravitational_sampler import GravitationalSampler

DIM = 8
VOCAB = 20
torch.manual_seed(42)


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def axis_vec(i, dim=DIM):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def make_store():
    return ContextBodyStore(embedding_dim=DIM, decay_interval=9999)


def make_sampler(G=1.0, **kwargs):
    store = make_store()
    return GravitationalSampler(body_store=store, G=G, **kwargs)


def make_body(centroid, mass):
    b = ContextBody()
    b.centroid = centroid.copy()
    b.mass = mass
    return b


def make_record(centroid, mass, last_seen=None):
    rec = ContextBodyRecord(
        centroid=centroid.copy().astype(float),
        mass=mass,
        last_seen=last_seen or datetime.utcnow(),
    )
    return rec


def make_logits(vocab=VOCAB):
    torch.manual_seed(0)
    return torch.zeros(vocab)


def make_token_embeddings(vocab=VOCAB, dim=DIM):
    torch.manual_seed(1)
    embs = torch.randn(vocab, dim)
    embs = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
    return embs


# ---------------------------------------------------------------------------
# _gravitational_force()
# ---------------------------------------------------------------------------

class TestGravitationalForce:

    def test_force_magnitude_formula(self):
        sampler = make_sampler(G=1.0)
        tok_emb = axis_vec(0).astype(float)
        body_centroid = norm(axis_vec(0) + axis_vec(1) * 0.5)
        tok_mass = 1.0
        body_mass = 4.0
        force = sampler._gravitational_force(tok_emb, tok_mass, body_centroid, body_mass)
        # r = cosine distance between tok and centroid
        r = float(1.0 - np.dot(norm(tok_emb), norm(body_centroid)))
        r = max(r, 1e-6)
        expected_magnitude = 1.0 * 1.0 * 4.0 / (r ** 2)
        actual_magnitude = float(np.linalg.norm(force))
        assert actual_magnitude == pytest.approx(expected_magnitude, rel=0.05)

    def test_force_points_toward_body(self):
        sampler = make_sampler(G=1.0)
        tok_emb = axis_vec(0).astype(float)
        body_centroid = norm(axis_vec(0) + axis_vec(1)).astype(float)
        force = sampler._gravitational_force(tok_emb, 1.0, body_centroid, 1.0)
        direction = body_centroid - tok_emb
        # Force should have positive dot product with direction toward body
        assert float(np.dot(force, direction)) > 0.0

    def test_doubling_body_mass_doubles_force(self):
        sampler = make_sampler(G=1.0)
        tok_emb = axis_vec(0).astype(float)
        body = norm(axis_vec(0) + axis_vec(1) * 0.3).astype(float)
        f1 = sampler._gravitational_force(tok_emb, 1.0, body, 2.0)
        f2 = sampler._gravitational_force(tok_emb, 1.0, body, 4.0)
        ratio = np.linalg.norm(f2) / (np.linalg.norm(f1) + 1e-10)
        assert ratio == pytest.approx(2.0, rel=0.05)

    def test_force_near_zero_distance_no_nan_or_inf(self):
        sampler = make_sampler(G=1.0)
        tok_emb = axis_vec(0).astype(float)
        body = axis_vec(0).astype(float)  # identical vector -> r near 0
        force = sampler._gravitational_force(tok_emb, 1.0, body, 1.0)
        assert not np.any(np.isnan(force))
        assert not np.any(np.isinf(force))


# ---------------------------------------------------------------------------
# _recency_factor()
# ---------------------------------------------------------------------------

class TestRecencyFactor:

    def test_context_body_always_returns_one(self):
        sampler = make_sampler(recency_decay_lambda=1e-4)
        body = make_body(axis_vec(0).astype(float), mass=1.0)
        assert sampler._recency_factor(body) == pytest.approx(1.0)

    def test_lambda_zero_always_returns_one(self):
        sampler = make_sampler(recency_decay_lambda=0.0)
        rec = make_record(axis_vec(0).astype(float), mass=1.0,
                          last_seen=datetime.utcnow() - timedelta(hours=5))
        assert sampler._recency_factor(rec) == pytest.approx(1.0)

    def test_fresh_record_factor_near_one(self):
        sampler = make_sampler(recency_decay_lambda=1e-4)
        rec = make_record(axis_vec(0).astype(float), mass=1.0,
                          last_seen=datetime.utcnow())
        factor = sampler._recency_factor(rec)
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_old_record_factor_less_than_one(self):
        sampler = make_sampler(recency_decay_lambda=1e-4)
        rec = make_record(axis_vec(0).astype(float), mass=1.0,
                          last_seen=datetime.utcnow() - timedelta(hours=1))
        factor = sampler._recency_factor(rec)
        expected = np.exp(-1e-4 * 3600)
        assert factor == pytest.approx(expected, rel=0.05)


# ---------------------------------------------------------------------------
# _group_active_bodies()
# ---------------------------------------------------------------------------

class TestGroupActiveBodies:

    def test_empty_active_bodies_returns_empty(self):
        sampler = make_sampler()
        sampler.active_bodies = []
        assert sampler._group_active_bodies() == []

    def test_single_body_returns_one_group(self):
        sampler = make_sampler(amplification_threshold=0.2)
        body = make_body(axis_vec(0).astype(float), mass=3.0)
        sampler.active_bodies = [(0, body, 0.1)]
        groups = sampler._group_active_bodies()
        assert len(groups) == 1

    def test_close_bodies_merged_into_one_group(self):
        sampler = make_sampler(amplification_threshold=0.3)
        # Two bodies very close (along same axis)
        b1 = make_body(norm(axis_vec(0) + axis_vec(1) * 0.01).astype(float), mass=2.0)
        b2 = make_body(norm(axis_vec(0) + axis_vec(1) * 0.02).astype(float), mass=3.0)
        sampler.active_bodies = [(0, b1, 0.1), (1, b2, 0.1)]
        groups = sampler._group_active_bodies()
        assert len(groups) == 1

    def test_close_bodies_merged_mass_is_sum(self):
        sampler = make_sampler(amplification_threshold=0.5)
        b1 = make_body(norm(axis_vec(0) + axis_vec(1) * 0.01).astype(float), mass=2.0)
        b2 = make_body(norm(axis_vec(0) + axis_vec(1) * 0.02).astype(float), mass=3.0)
        sampler.active_bodies = [(0, b1, 0.1), (1, b2, 0.1)]
        groups = sampler._group_active_bodies()
        total_mass = sum(m for _, m in groups)
        assert total_mass == pytest.approx(5.0, rel=0.05)

    def test_far_bodies_stay_separate(self):
        sampler = make_sampler(amplification_threshold=0.1)
        # Orthogonal axes -> cosine dist = 1.0 >> threshold
        b1 = make_body(axis_vec(0).astype(float), mass=1.0)
        b2 = make_body(axis_vec(1).astype(float), mass=1.0)
        sampler.active_bodies = [(0, b1, 0.1), (1, b2, 0.1)]
        groups = sampler._group_active_bodies()
        assert len(groups) == 2

    def test_recency_reduces_effective_mass_in_group(self):
        sampler = make_sampler(amplification_threshold=0.5, recency_decay_lambda=1e-3)
        # ContextBodyRecord with very old last_seen -> low recency factor
        rec = make_record(
            norm(axis_vec(0) + axis_vec(1) * 0.01).astype(float),
            mass=10.0,
            last_seen=datetime.utcnow() - timedelta(hours=24),
        )
        sampler.active_bodies = [(-1, rec, 0.1)]
        groups = sampler._group_active_bodies()
        # Effective mass = mass * recency_factor < mass
        _, eff_mass = groups[0]
        assert eff_mass < 10.0


# ---------------------------------------------------------------------------
# _compute_resonance_forces()
# ---------------------------------------------------------------------------

class TestComputeResonanceForces:

    def test_no_records_gives_zero_force(self):
        sampler = make_sampler(resonance_threshold=0.3)
        body = make_body(axis_vec(0).astype(float), mass=2.0)
        sampler.active_bodies = [(0, body, 0.1)]  # only ContextBody, no records
        force = sampler._compute_resonance_forces(axis_vec(0).astype(float), 1.0)
        assert np.allclose(force, 0.0)

    def test_low_score_gives_zero_force(self):
        sampler = make_sampler(resonance_threshold=0.5)
        rec_a = make_record(axis_vec(0).astype(float), mass=1.0)
        rec_b = make_record(axis_vec(1).astype(float), mass=1.0)
        # score below threshold
        rec_a.resonance_partners[str(rec_b.id)] = 0.1
        sampler.active_bodies = [(-1, rec_a, 0.1), (-1, rec_b, 0.1)]
        force = sampler._compute_resonance_forces(axis_vec(2).astype(float), 1.0)
        assert np.allclose(force, 0.0)

    def test_high_score_gives_nonzero_force(self):
        sampler = make_sampler(resonance_threshold=0.3)
        rec_a = make_record(axis_vec(0).astype(float), mass=5.0)
        rec_b = make_record(axis_vec(1).astype(float), mass=5.0)
        rec_a.resonance_partners[str(rec_b.id)] = 0.9
        sampler.active_bodies = [(-1, rec_a, 0.1), (-1, rec_b, 0.1)]
        tok = norm(axis_vec(3)).astype(float)  # different direction
        force = sampler._compute_resonance_forces(tok, 1.0)
        assert np.linalg.norm(force) > 0.0

    def test_force_scales_with_joint_mass(self):
        sampler = make_sampler(resonance_threshold=0.3, G=1.0)
        tok = norm(axis_vec(3)).astype(float)

        rec_a1 = make_record(axis_vec(0).astype(float), mass=1.0)
        rec_b1 = make_record(axis_vec(1).astype(float), mass=1.0)
        rec_a1.resonance_partners[str(rec_b1.id)] = 1.0
        sampler.active_bodies = [(-1, rec_a1, 0.1), (-1, rec_b1, 0.1)]
        f1 = np.linalg.norm(sampler._compute_resonance_forces(tok, 1.0))

        rec_a2 = make_record(axis_vec(0).astype(float), mass=4.0)
        rec_b2 = make_record(axis_vec(1).astype(float), mass=4.0)
        rec_a2.resonance_partners[str(rec_b2.id)] = 1.0
        sampler.active_bodies = [(-1, rec_a2, 0.1), (-1, rec_b2, 0.1)]
        f2 = np.linalg.norm(sampler._compute_resonance_forces(tok, 1.0))

        # joint_mass = sqrt(m_A * m_B) * score; 4x heavier -> 4x joint_mass
        assert f2 > f1


# ---------------------------------------------------------------------------
# _check_collisions()
# ---------------------------------------------------------------------------

class TestCheckCollisions:

    def test_close_bodies_merged(self):
        sampler = make_sampler(collision_distance=0.5)
        b1 = make_body(norm(axis_vec(0) + axis_vec(1) * 0.01).astype(float), mass=2.0)
        b2 = make_body(norm(axis_vec(0) + axis_vec(1) * 0.02).astype(float), mass=3.0)
        sampler.active_bodies = [(0, b1, 0.1), (1, b2, 0.1)]
        sampler._check_collisions()
        assert len(sampler.active_bodies) == 1

    def test_merged_mass_is_sum(self):
        sampler = make_sampler(collision_distance=0.5)
        b1 = make_body(norm(axis_vec(0) + axis_vec(1) * 0.01).astype(float), mass=2.0)
        b2 = make_body(norm(axis_vec(0) + axis_vec(1) * 0.02).astype(float), mass=3.0)
        sampler.active_bodies = [(0, b1, 0.1), (1, b2, 0.1)]
        sampler._check_collisions()
        _, merged_body, _ = sampler.active_bodies[0]
        assert merged_body.mass == pytest.approx(5.0)

    def test_far_bodies_not_merged(self):
        sampler = make_sampler(collision_distance=0.1)
        b1 = make_body(axis_vec(0).astype(float), mass=1.0)
        b2 = make_body(axis_vec(1).astype(float), mass=1.0)  # orthogonal -> dist=1.0
        sampler.active_bodies = [(0, b1, 0.1), (1, b2, 0.1)]
        sampler._check_collisions()
        assert len(sampler.active_bodies) == 2

    def test_records_do_not_participate_in_collision(self):
        sampler = make_sampler(collision_distance=0.9)
        rec = make_record(axis_vec(0).astype(float), mass=1.0)
        body = make_body(norm(axis_vec(0) + axis_vec(1) * 0.01).astype(float), mass=1.0)
        # label=-1 for record, label>=0 for body
        sampler.active_bodies = [(-1, rec, 0.1), (0, body, 0.1)]
        sampler._check_collisions()
        # records never merge with each other or with bodies
        assert len(sampler.active_bodies) == 2


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------

class TestSample:

    def test_returns_integer_in_range(self):
        sampler = make_sampler(G=1.0)
        sampler.active_bodies = []
        logits = make_logits()
        embs = make_token_embeddings()
        idx = sampler.sample(logits, embs)
        assert isinstance(idx, int)
        assert 0 <= idx < VOCAB

    def test_g_zero_still_returns_valid_token(self):
        sampler = make_sampler(G=0.0)
        sampler.active_bodies = []
        idx = sampler.sample(make_logits(), make_token_embeddings())
        assert 0 <= idx < VOCAB

    def test_strong_body_biases_toward_its_direction(self):
        # Body with huge mass along axis 0. Tokens near axis 0 should dominate.
        sampler = make_sampler(G=100.0)
        heavy_body = make_body(axis_vec(0).astype(float), mass=1000.0)
        sampler.active_bodies = [(0, heavy_body, 0.01)]

        # Token embeddings: token 0 is along axis 0, rest are orthogonal or random
        embs = torch.zeros(VOCAB, DIM)
        embs[0] = torch.tensor(axis_vec(0))  # perfectly aligned with body
        for i in range(1, VOCAB):
            v = torch.zeros(DIM); v[1] = 1.0  # orthogonal to body
            embs[i] = v

        logits = torch.zeros(VOCAB)
        counts = {0: 0, "other": 0}
        for _ in range(100):
            idx = sampler.sample(logits, embs)
            if idx == 0:
                counts[0] += 1
            else:
                counts["other"] += 1
        # With very strong gravity along axis 0, token 0 should win most samples
        assert counts[0] > 50

    def test_escape_rate_tracking(self):
        sampler = make_sampler(G=0.0, escape_threshold=0.01)
        sampler.active_bodies = []
        sampler.sample(make_logits(), make_token_embeddings())
        # With G=0 all forces are 0 -> all tokens are below escape_threshold
        assert sampler._last_escape_count == VOCAB
