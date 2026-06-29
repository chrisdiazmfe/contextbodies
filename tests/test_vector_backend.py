import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from vector_backend import FAISSBackend

DIM = 8
rng = np.random.default_rng(42)


def rand_vec(seed=None):
    r = np.random.default_rng(seed)
    v = r.standard_normal(DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# upsert + search
# ---------------------------------------------------------------------------

class TestUpsertAndSearch:

    def test_search_finds_inserted_vector(self):
        be = FAISSBackend(DIM)
        v = rand_vec(1)
        be.upsert("id-1", v, {"domain": "test"})
        results = be.search(v, k=1)
        assert len(results) == 1
        assert results[0][0] == "id-1"

    def test_search_result_is_four_tuple(self):
        be = FAISSBackend(DIM)
        v = rand_vec(2)
        be.upsert("id-2", v, {})
        result = be.search(v, k=1)[0]
        rec_id, dist, meta, stored_vec = result  # must unpack as 4-tuple
        assert isinstance(rec_id, str)
        assert isinstance(dist, float)
        assert isinstance(meta, dict)
        assert stored_vec is None  # FAISSBackend always returns None

    def test_nearest_neighbor_ranked_first(self):
        be = FAISSBackend(DIM)
        near = rand_vec(10)
        far = np.zeros(DIM, dtype=np.float32)
        far[1] = 1.0  # orthogonal to near (large cosine distance)
        be.upsert("near", near, {})
        be.upsert("far", far, {})
        results = be.search(near, k=2)
        assert results[0][0] == "near"

    def test_distance_is_ascending(self):
        be = FAISSBackend(DIM)
        v0 = rand_vec(20)
        v1 = rand_vec(21)
        v2 = rand_vec(22)
        for rid, v in [("a", v0), ("b", v1), ("c", v2)]:
            be.upsert(rid, v, {})
        results = be.search(v0, k=3)
        dists = [r[1] for r in results]
        assert dists == sorted(dists)

    def test_empty_store_returns_empty_list(self):
        be = FAISSBackend(DIM)
        results = be.search(rand_vec(99), k=5)
        assert results == []


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:

    def test_delete_removes_from_search_results(self):
        be = FAISSBackend(DIM)
        v = rand_vec(3)
        be.upsert("del-me", v, {})
        be.delete(["del-me"])
        results = be.search(v, k=5)
        ids = [r[0] for r in results]
        assert "del-me" not in ids

    def test_size_decreases_after_delete(self):
        be = FAISSBackend(DIM)
        be.upsert("a", rand_vec(4), {})
        be.upsert("b", rand_vec(5), {})
        assert be.size == 2
        be.delete(["a"])
        assert be.size == 1

    def test_delete_nonexistent_does_not_crash(self):
        be = FAISSBackend(DIM)
        be.delete(["nonexistent-id"])  # should not raise


# ---------------------------------------------------------------------------
# update_metadata
# ---------------------------------------------------------------------------

class TestUpdateMetadata:

    def test_update_metadata_changes_field(self):
        be = FAISSBackend(DIM)
        v = rand_vec(6)
        be.upsert("meta-test", v, {"mass": 1.0, "domain": "a"})
        be.update_metadata("meta-test", {"mass": 5.0})
        results = be.search(v, k=1)
        assert results[0][2]["mass"] == pytest.approx(5.0)

    def test_update_metadata_preserves_other_fields(self):
        be = FAISSBackend(DIM)
        v = rand_vec(7)
        be.upsert("meta-preserve", v, {"mass": 1.0, "domain": "keep-me"})
        be.update_metadata("meta-preserve", {"mass": 9.0})
        results = be.search(v, k=1)
        assert results[0][2]["domain"] == "keep-me"

    def test_update_metadata_does_not_remove_vector(self):
        be = FAISSBackend(DIM)
        v = rand_vec(8)
        be.upsert("still-there", v, {})
        be.update_metadata("still-there", {"extra": "value"})
        assert be.size == 1
        results = be.search(v, k=1)
        assert results[0][0] == "still-there"


# ---------------------------------------------------------------------------
# size property
# ---------------------------------------------------------------------------

class TestSize:

    def test_size_starts_at_zero(self):
        be = FAISSBackend(DIM)
        assert be.size == 0

    def test_size_increments_on_upsert(self):
        be = FAISSBackend(DIM)
        be.upsert("x1", rand_vec(30), {})
        assert be.size == 1
        be.upsert("x2", rand_vec(31), {})
        assert be.size == 2

    def test_upsert_same_id_does_not_increase_size(self):
        be = FAISSBackend(DIM)
        v = rand_vec(40)
        be.upsert("dup", v, {"v": 1})
        be.upsert("dup", v, {"v": 2})  # same id
        assert be.size == 1


# ---------------------------------------------------------------------------
# filter support
# ---------------------------------------------------------------------------

class TestFilter:

    def test_filter_excludes_different_domain(self):
        be = FAISSBackend(DIM)
        v = rand_vec(50)
        be.upsert("code-body", v, {"domain": "code"})
        # Search near same vector but filter for "math" — should get nothing
        be.upsert("math-body", rand_vec(51), {"domain": "math"})
        results = be.search(v, k=5, filter={"domain": "math"})
        ids = [r[0] for r in results]
        assert "code-body" not in ids

    def test_filter_includes_matching_domain(self):
        be = FAISSBackend(DIM)
        v = rand_vec(52)
        be.upsert("code-1", v, {"domain": "code"})
        be.upsert("other-1", rand_vec(53), {"domain": "other"})
        results = be.search(v, k=5, filter={"domain": "code"})
        ids = [r[0] for r in results]
        assert "code-1" in ids
        assert "other-1" not in ids
