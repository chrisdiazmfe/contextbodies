import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from domain_classifier import DomainClassifier

DIM = 8


def axis_vec(i, dim=DIM):
    v = np.zeros(dim, dtype=float)
    v[i] = 1.0
    return v


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


# axis-aligned unit vectors for each domain — maximally distinct in cosine space
CODE_ANCHOR    = axis_vec(0)
MEDICAL_ANCHOR = axis_vec(1)
LEGAL_ANCHOR   = axis_vec(2)
MATH_ANCHOR    = axis_vec(3)


def make_clf(**kwargs):
    anchors = {
        "code":    CODE_ANCHOR.copy(),
        "medical": MEDICAL_ANCHOR.copy(),
        "legal":   LEGAL_ANCHOR.copy(),
    }
    defaults = dict(match_threshold=0.5, fallback_domain="general")
    defaults.update(kwargs)
    return DomainClassifier(anchors, **defaults)


# ---------------------------------------------------------------------------
# context_direction before seeding
# ---------------------------------------------------------------------------

class TestContextDirectionInit:

    def test_context_direction_is_none_before_seed(self):
        clf = make_clf()
        assert clf.context_direction is None

    def test_context_direction_is_none_before_update(self):
        clf = DomainClassifier({})
        assert clf.context_direction is None


# ---------------------------------------------------------------------------
# seed()
# ---------------------------------------------------------------------------

class TestSeed:

    def test_seed_sets_context_direction(self):
        clf = make_clf()
        embeddings = np.tile(CODE_ANCHOR, (5, 1))  # [5, DIM]
        clf.seed(embeddings)
        assert clf.context_direction is not None

    def test_seed_classifies_code_domain(self):
        clf = make_clf()
        embeddings = np.tile(CODE_ANCHOR, (5, 1))
        domain = clf.seed(embeddings)
        assert domain == "code"

    def test_seed_classifies_medical_domain(self):
        clf = make_clf()
        embeddings = np.tile(MEDICAL_ANCHOR, (5, 1))
        domain = clf.seed(embeddings)
        assert domain == "medical"

    def test_seed_context_direction_is_unit_norm(self):
        clf = make_clf()
        embeddings = np.tile(CODE_ANCHOR, (3, 1))
        clf.seed(embeddings)
        assert abs(np.linalg.norm(clf.context_direction) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:

    def test_update_returns_string(self):
        clf = make_clf()
        domain = clf.update(CODE_ANCHOR)
        assert isinstance(domain, str)

    def test_update_classifies_toward_code(self):
        clf = make_clf(ema_alpha=1.0)  # fully reactive
        domain = clf.update(CODE_ANCHOR)
        assert domain == "code"

    def test_update_sets_context_direction(self):
        clf = make_clf()
        clf.update(CODE_ANCHOR)
        assert clf.context_direction is not None

    def test_ema_keeps_domain_stable_under_sustained_input(self):
        clf = make_clf(ema_alpha=0.1, match_threshold=0.5)
        for _ in range(30):
            clf.update(CODE_ANCHOR + np.random.default_rng(99).standard_normal(DIM) * 0.01)
        assert clf.current_domain == "code"


# ---------------------------------------------------------------------------
# _classify() fallback behavior
# ---------------------------------------------------------------------------

class TestClassifyFallback:

    def test_fallback_when_no_anchors(self):
        clf = DomainClassifier({}, fallback_domain="general")
        clf._context_direction = norm(CODE_ANCHOR)
        assert clf._classify() == "general"

    def test_fallback_when_context_direction_is_none(self):
        clf = make_clf()
        # Don't call seed or update — context_direction stays None
        assert clf._classify() == "general"

    def test_fallback_when_nearest_anchor_beyond_threshold(self):
        # Put the anchor far from our query direction
        anchor = axis_vec(0)  # code is along dim 0
        query = axis_vec(7)   # orthogonal — cosine dist = 1.0
        clf = DomainClassifier({"code": anchor}, match_threshold=0.3, fallback_domain="general")
        clf._context_direction = norm(query)
        assert clf._classify() == "general"

    def test_classifies_nearest_anchor_within_threshold(self):
        clf = make_clf(match_threshold=0.5)
        clf._context_direction = norm(CODE_ANCHOR)
        assert clf._classify() == "code"


# ---------------------------------------------------------------------------
# add_anchor() / remove_anchor()
# ---------------------------------------------------------------------------

class TestAddRemoveAnchor:

    def test_add_anchor_enables_classification(self):
        clf = DomainClassifier({}, match_threshold=0.5, fallback_domain="general")
        clf.add_anchor("math", MATH_ANCHOR.copy())
        clf._context_direction = norm(MATH_ANCHOR)
        assert clf._classify() == "math"

    def test_add_anchor_normalizes_embedding(self):
        clf = DomainClassifier({})
        big = MATH_ANCHOR * 100.0
        clf.add_anchor("big", big)
        stored = clf.domain_anchors["big"]
        assert abs(np.linalg.norm(stored) - 1.0) < 1e-5

    def test_remove_anchor_removes_domain(self):
        clf = make_clf()
        clf.remove_anchor("code")
        assert "code" not in clf.domain_anchors

    def test_remove_anchor_reclassifies_if_was_current(self):
        clf = make_clf(match_threshold=0.5)
        clf._context_direction = norm(CODE_ANCHOR)
        clf._current_domain = "code"
        clf.remove_anchor("code")
        # After removal, current domain must no longer be "code"
        assert clf.current_domain != "code"

    def test_remove_nonexistent_anchor_does_not_crash(self):
        clf = make_clf()
        clf.remove_anchor("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# from_body_centroids()
# ---------------------------------------------------------------------------

class TestFromBodyCentroids:

    def test_builds_anchor_as_mean_of_centroids(self):
        c1 = axis_vec(0)
        c2 = axis_vec(0) * 0.9 + axis_vec(1) * 0.1
        bodies_by_domain = {"code": [c1, c2]}
        clf = DomainClassifier.from_body_centroids(
            bodies_by_domain, match_threshold=0.5, fallback_domain="general"
        )
        # anchor should exist for "code"
        assert "code" in clf.domain_anchors

    def test_classifies_correctly_from_centroids(self):
        bodies_by_domain = {
            "code":    [axis_vec(0), axis_vec(0)],
            "medical": [axis_vec(1), axis_vec(1)],
        }
        clf = DomainClassifier.from_body_centroids(
            bodies_by_domain, match_threshold=0.5, fallback_domain="general"
        )
        clf._context_direction = norm(axis_vec(0))
        assert clf._classify() == "code"

    def test_empty_domain_list_excluded(self):
        bodies_by_domain = {"code": [], "medical": [axis_vec(1)]}
        clf = DomainClassifier.from_body_centroids(bodies_by_domain)
        assert "code" not in clf.domain_anchors
        assert "medical" in clf.domain_anchors
