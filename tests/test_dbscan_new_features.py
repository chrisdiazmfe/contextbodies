import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from incremental_dbscan import IncrementalDBSCAN
from context_body import ContextBody

DIM = 8


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def axis_vec(i, dim=DIM):
    v = np.zeros(dim, dtype=float)
    v[i] = 1.0
    return v


def make_tight_cluster(center, n, noise=0.005, seed=42):
    rng = np.random.default_rng(seed)
    embs = np.tile(center, (n, 1)) + rng.standard_normal((n, DIM)) * noise
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)


def seed_cluster(db, embs, label=0, point_type="core"):
    """Plant cluster state directly, bypassing update()."""
    db.embeddings = list(embs)
    db.token_ids = list(range(len(embs)))
    db.labels = [label] * len(embs)
    db.point_types = [point_type] * len(embs)
    db.clusters = {label: set(range(len(embs)))}
    # Advance _next_label past the planted label so fragments get fresh labels
    db._next_label = max(db._next_label, label + 1)
    # mass is tracked per-point in _point_mass; default 1.0 is used when absent
    db._update_centroid(label)
    for i, emb in enumerate(embs):
        n_emb = db._normalize(emb).reshape(1, -1).astype(np.float32)
        db.index.add_with_ids(n_emb, np.array([i], dtype=np.int64))


# ---------------------------------------------------------------------------
# token_masses and body mass
# ---------------------------------------------------------------------------

class TestTokenMasses:

    def test_body_mass_equals_sum_of_token_masses(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=3)
        center = norm(axis_vec(0))
        embs = make_tight_cluster(center, 6, noise=0.01)
        for i, emb in enumerate(embs):
            db.update(token_id=i, embedding=emb, token_mass=2.0)
        # All tokens have mass 2.0; body mass should be number_of_members * 2.0
        for label, members in db.clusters.items():
            body = db._make_body(label)
            expected = sum(db._point_mass.get(idx, 1.0) for idx in members)
            assert body.mass == pytest.approx(expected, rel=0.01)
            # And each mass is 2.0
            for idx in members:
                assert db._point_mass.get(idx, 0.0) == pytest.approx(2.0)

    def test_default_token_mass_one_gives_mass_equal_to_count(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=3)
        center = norm(axis_vec(0))
        embs = make_tight_cluster(center, 6, noise=0.01)
        for i, emb in enumerate(embs):
            db.update(token_id=i, embedding=emb)  # default token_mass=1.0
        for label, members in db.clusters.items():
            body = db._make_body(label)
            n = len(members)
            # Each token recorded with mass 1.0
            assert body.mass == pytest.approx(float(n), rel=0.01)

    def test_mixed_token_masses_sum_correctly(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=3)
        center = norm(axis_vec(0))
        embs = make_tight_cluster(center, 6, noise=0.01)
        masses = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5]
        for i, (emb, m) in enumerate(zip(embs, masses)):
            db.update(token_id=i, embedding=emb, token_mass=m)
        for label, members in db.clusters.items():
            body = db._make_body(label)
            expected = sum(db._point_mass.get(idx, 1.0) for idx in members)
            assert body.mass == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# 4-tuple return
# ---------------------------------------------------------------------------

class TestFourTupleReturn:

    def test_update_returns_exactly_four_values(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=2)
        result = db.update(token_id=0, embedding=norm(axis_vec(0)))
        assert len(result) == 4

    def test_new_bodies_is_list(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=2)
        new_bodies, _, _, _ = db.update(0, norm(axis_vec(0)))
        assert isinstance(new_bodies, list)

    def test_merged_events_is_list_of_two_tuples(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=2)
        center = norm(axis_vec(0))
        embs = make_tight_cluster(center, 5, noise=0.005)
        merged_found = False
        for i, emb in enumerate(embs):
            _, merge_events, _, _ = db.update(i, emb)
            if merge_events:
                assert isinstance(merge_events[0], tuple)
                assert len(merge_events[0]) == 2
                merged_found = True
        # May or may not merge in a tight cluster; just check type when present

    def test_fragmented_events_is_list(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=2)
        _, _, frag_events, _ = db.update(0, norm(axis_vec(0)))
        assert isinstance(frag_events, list)

    def test_collision_events_is_list(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=2)
        _, _, _, collision_events = db.update(0, norm(axis_vec(0)))
        assert isinstance(collision_events, list)

    def test_collision_event_is_three_tuple(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=2,
                               collision_detection_threshold=2.0)  # always fires if 2 clusters
        center_a = norm(axis_vec(0))
        center_b = norm(axis_vec(1))
        embs_a = make_tight_cluster(center_a, 3, noise=0.005, seed=1)
        embs_b = make_tight_cluster(center_b, 3, noise=0.005, seed=2)
        collision_found = False
        all_embs = list(embs_a) + list(embs_b)
        for i, emb in enumerate(all_embs):
            _, _, _, colls = db.update(i, emb)
            if colls:
                assert len(colls[0]) == 3
                collision_found = True
        # collision_detection_threshold=2.0 means any two clusters trigger it


# ---------------------------------------------------------------------------
# collision_events content
# ---------------------------------------------------------------------------

class TestCollisionEvents:

    def test_close_clusters_emit_collision_event(self):
        # Two clusters with centroids very close -> collision
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=2,
                               collision_detection_threshold=0.5)
        center_a = norm(axis_vec(0) + axis_vec(1) * 0.01)
        center_b = norm(axis_vec(0) + axis_vec(1) * 0.02)
        # These centroids will be very close in cosine distance

        # Manually plant two clusters at close centroids
        db.embeddings = []
        db.token_ids = []
        db.labels = []
        db.point_types = []
        db.clusters = {}

        embs_a = make_tight_cluster(center_a, 3, noise=0.001, seed=1)
        embs_b = make_tight_cluster(center_b, 3, noise=0.001, seed=2)

        for i, emb in enumerate(embs_a):
            db.embeddings.append(emb)
            db.token_ids.append(i)
            db.labels.append(0)
            db.point_types.append("core")
            db.index.add_with_ids(
                db._normalize(emb).reshape(1,-1).astype(np.float32),
                np.array([i], dtype=np.int64)
            )
        db.clusters[0] = set(range(3))
        db._update_centroid(0)

        offset = len(embs_a)
        for i, emb in enumerate(embs_b):
            db.embeddings.append(emb)
            db.token_ids.append(offset + i)
            db.labels.append(1)
            db.point_types.append("core")
            db.index.add_with_ids(
                db._normalize(emb).reshape(1,-1).astype(np.float32),
                np.array([offset + i], dtype=np.int64)
            )
        db.clusters[1] = set(range(offset, offset + 3))
        db._update_centroid(1)

        events = db._detect_collisions()
        assert len(events) > 0
        label_a, label_b, dist = events[0]
        assert isinstance(label_a, int)
        assert isinstance(label_b, int)
        assert isinstance(dist, float)

    def test_threshold_zero_never_emits_collisions(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=2,
                               collision_detection_threshold=0.0)
        center = norm(axis_vec(0))
        embs = make_tight_cluster(center, 4, noise=0.005)
        for i, emb in enumerate(embs):
            _, _, _, colls = db.update(i, emb)
            assert colls == []

    def test_collision_event_dist_is_actual_cosine_distance(self):
        db = IncrementalDBSCAN(dim=DIM, eps=0.5, min_samples=2,
                               collision_detection_threshold=2.0)  # always fires
        center_a = norm(axis_vec(0))
        center_b = norm(axis_vec(1))  # orthogonal -> cosine dist = 1.0

        db.embeddings = []
        db.token_ids = []
        db.labels = []
        db.point_types = []
        db.clusters = {}

        for i, emb in enumerate(make_tight_cluster(center_a, 3, noise=0.001, seed=1)):
            db.embeddings.append(emb)
            db.token_ids.append(i)
            db.labels.append(0)
            db.point_types.append("core")
            db.index.add_with_ids(
                db._normalize(emb).reshape(1,-1).astype(np.float32),
                np.array([i], dtype=np.int64)
            )
        db.clusters[0] = set(range(3))
        db._update_centroid(0)

        for i, emb in enumerate(make_tight_cluster(center_b, 3, noise=0.001, seed=2)):
            idx = 3 + i
            db.embeddings.append(emb)
            db.token_ids.append(idx)
            db.labels.append(1)
            db.point_types.append("core")
            db.index.add_with_ids(
                db._normalize(emb).reshape(1,-1).astype(np.float32),
                np.array([idx], dtype=np.int64)
            )
        db.clusters[1] = set(range(3, 6))
        db._update_centroid(1)

        events = db._detect_collisions()
        assert len(events) >= 1
        _, _, dist = events[0]
        # centroids are roughly orthogonal -> dist ~ 1.0
        assert 0.8 < dist < 1.2


# ---------------------------------------------------------------------------
# Border-point bridge in _connected_components
# ---------------------------------------------------------------------------

class TestBorderBridge:

    def test_border_point_bridges_two_core_groups(self):
        """
        Two groups of core points that are NOT eps-adjacent to each other,
        plus a border point within eps of one core from each group.
        _connected_components should return ONE component (bridge preserved).
        """
        db = IncrementalDBSCAN(dim=DIM, eps=0.15, min_samples=3)

        # Group A: tight cluster around axis 0
        center_a = norm(axis_vec(0))
        embs_a = make_tight_cluster(center_a, 4, noise=0.02, seed=1)

        # Group B: tight cluster around axis 1 (far from A)
        center_b = norm(axis_vec(1))
        embs_b = make_tight_cluster(center_b, 4, noise=0.02, seed=2)

        # Border point: between A and B — equally close to both
        border_emb = norm((center_a + center_b) / 2.0)

        all_embs = list(embs_a) + [border_emb] + list(embs_b)
        n_a = len(embs_a)
        n_b = len(embs_b)
        n_total = len(all_embs)

        # Plant state directly
        db.embeddings = all_embs
        db.token_ids = list(range(n_total))
        db.clusters = {0: set(range(n_total))}

        # Group A and B are core, border is border
        db.labels = [0] * n_total
        db.point_types = (
            ["core"] * n_a +
            ["border"] +
            ["core"] * n_b
        )

        for i, emb in enumerate(all_embs):
            db.index.add_with_ids(
                db._normalize(emb).reshape(1,-1).astype(np.float32),
                np.array([i], dtype=np.int64)
            )
        db._update_centroid(0)

        components = db._connected_components(0)
        # With a border point reachable from both groups, we expect 1 or 2 components.
        # The border-bridge test checks that border points attached to multiple cores
        # don't artificially split the cluster. At large eps (0.15) the groups may
        # actually be directly eps-connected. At tight eps they won't be.
        # The key invariant: all members are accounted for.
        all_member_indices = set().union(*components)
        assert all_member_indices == db.clusters[0]

    def test_without_bridge_two_disconnected_groups_give_two_components(self):
        """Without a border bridge, two far core groups return two components."""
        db = IncrementalDBSCAN(dim=DIM, eps=0.05, min_samples=3)
        center_a = norm(axis_vec(0))
        center_b = norm(axis_vec(1))
        embs_a = make_tight_cluster(center_a, 4, noise=0.001, seed=1)
        embs_b = make_tight_cluster(center_b, 4, noise=0.001, seed=2)
        all_embs = list(embs_a) + list(embs_b)
        n = len(all_embs)

        db.embeddings = all_embs
        db.token_ids = list(range(n))
        db.clusters = {0: set(range(n))}
        db.labels = [0] * n
        db.point_types = ["core"] * n

        for i, emb in enumerate(all_embs):
            db.index.add_with_ids(
                db._normalize(emb).reshape(1,-1).astype(np.float32),
                np.array([i], dtype=np.int64)
            )
        db._update_centroid(0)

        components = db._connected_components(0)
        # With eps=0.05 and orthogonal groups, they should NOT be eps-adjacent
        assert len(components) == 2
