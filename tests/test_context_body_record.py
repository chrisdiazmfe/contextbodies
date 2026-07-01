import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import pytest
from datetime import datetime, timezone
from uuid import UUID, uuid4
from context_body_record import ContextBodyRecord

DIM = 8


def make_centroid(dim=DIM):
    v = np.ones(dim, dtype=float)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# to_metadata()
# ---------------------------------------------------------------------------

class TestToMetadata:

    def test_returns_dict(self):
        rec = ContextBodyRecord(centroid=make_centroid(), mass=5.0, stability=0.8, domain="code")
        meta = rec.to_metadata()
        assert isinstance(meta, dict)

    def test_all_expected_keys_present(self):
        rec = ContextBodyRecord(centroid=make_centroid(), mass=5.0, stability=0.8, domain="code")
        meta = rec.to_metadata()
        for key in ("id", "mass", "stability", "domain", "created_at", "last_seen", "resonance_partners"):
            assert key in meta, f"missing key: {key}"

    def test_id_is_string(self):
        rec = ContextBodyRecord(centroid=make_centroid())
        meta = rec.to_metadata()
        assert isinstance(meta["id"], str)
        UUID(meta["id"])  # must be valid UUID string

    def test_mass_and_stability_are_floats(self):
        rec = ContextBodyRecord(centroid=make_centroid(), mass=3.14, stability=0.5)
        meta = rec.to_metadata()
        assert meta["mass"] == pytest.approx(3.14)
        assert meta["stability"] == pytest.approx(0.5)

    def test_domain_stored_correctly(self):
        rec = ContextBodyRecord(centroid=make_centroid(), domain="medical")
        meta = rec.to_metadata()
        assert meta["domain"] == "medical"

    def test_resonance_partners_serialized_as_json_string(self):
        partners = {"abc-123": 0.7, "def-456": 0.3}
        rec = ContextBodyRecord(centroid=make_centroid(), resonance_partners=partners)
        meta = rec.to_metadata()
        assert isinstance(meta["resonance_partners"], str)
        parsed = json.loads(meta["resonance_partners"])
        assert parsed == partners

    def test_empty_resonance_partners_is_json_empty_object(self):
        rec = ContextBodyRecord(centroid=make_centroid())
        meta = rec.to_metadata()
        assert meta["resonance_partners"] == "{}"

    def test_created_at_and_last_seen_are_iso_strings(self):
        rec = ContextBodyRecord(centroid=make_centroid())
        meta = rec.to_metadata()
        # Should parse without error
        datetime.fromisoformat(meta["created_at"])
        datetime.fromisoformat(meta["last_seen"])


# ---------------------------------------------------------------------------
# from_metadata() round-trip
# ---------------------------------------------------------------------------

class TestFromMetadata:

    def _make_metadata(self, **kwargs):
        rec = ContextBodyRecord(centroid=make_centroid(), **kwargs)
        return make_centroid(), rec.to_metadata(), rec

    def test_round_trip_id(self):
        centroid, meta, original = self._make_metadata(mass=2.0)
        restored = ContextBodyRecord.from_metadata(centroid=centroid, metadata=meta)
        assert restored.id == original.id

    def test_round_trip_mass(self):
        centroid, meta, original = self._make_metadata(mass=7.5)
        restored = ContextBodyRecord.from_metadata(centroid=centroid, metadata=meta)
        assert restored.mass == pytest.approx(7.5)

    def test_round_trip_stability(self):
        centroid, meta, original = self._make_metadata(stability=0.93)
        restored = ContextBodyRecord.from_metadata(centroid=centroid, metadata=meta)
        assert restored.stability == pytest.approx(0.93)

    def test_round_trip_domain(self):
        centroid, meta, original = self._make_metadata(domain="legal")
        restored = ContextBodyRecord.from_metadata(centroid=centroid, metadata=meta)
        assert restored.domain == "legal"

    def test_round_trip_last_seen(self):
        ts = datetime(2025, 6, 1, 12, 0, 0)
        rec = ContextBodyRecord(centroid=make_centroid(), last_seen=ts)
        meta = rec.to_metadata()
        restored = ContextBodyRecord.from_metadata(centroid=make_centroid(), metadata=meta)
        assert restored.last_seen == ts

    def test_round_trip_resonance_partners(self):
        partners = {"uid-1": 0.9, "uid-2": 0.4}
        rec = ContextBodyRecord(centroid=make_centroid(), resonance_partners=partners)
        meta = rec.to_metadata()
        restored = ContextBodyRecord.from_metadata(centroid=make_centroid(), metadata=meta)
        assert restored.resonance_partners == partners

    def test_missing_resonance_partners_returns_empty_dict(self):
        centroid = make_centroid()
        meta = {
            "id": str(uuid4()),
            "mass": 1.0,
            "stability": 0.5,
            "domain": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            # "resonance_partners" intentionally absent
        }
        restored = ContextBodyRecord.from_metadata(centroid=centroid, metadata=meta)
        assert restored.resonance_partners == {}

    def test_malformed_resonance_partners_returns_empty_dict(self):
        centroid = make_centroid()
        meta = {
            "id": str(uuid4()),
            "mass": 1.0,
            "stability": 0.5,
            "domain": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "resonance_partners": "NOT VALID JSON {{{",
        }
        restored = ContextBodyRecord.from_metadata(centroid=centroid, metadata=meta)
        assert restored.resonance_partners == {}

    def test_missing_created_at_does_not_crash(self):
        centroid = make_centroid()
        meta = {
            "id": str(uuid4()),
            "mass": 1.0,
            "stability": 0.5,
            "domain": "",
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "resonance_partners": "{}",
            # "created_at" absent
        }
        restored = ContextBodyRecord.from_metadata(centroid=centroid, metadata=meta)
        assert isinstance(restored.created_at, datetime)

    def test_missing_last_seen_does_not_crash(self):
        centroid = make_centroid()
        meta = {
            "id": str(uuid4()),
            "mass": 1.0,
            "stability": 0.5,
            "domain": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "resonance_partners": "{}",
            # "last_seen" absent
        }
        restored = ContextBodyRecord.from_metadata(centroid=centroid, metadata=meta)
        assert isinstance(restored.last_seen, datetime)
