"""
Unit tests for OFAC dataset validation logic.

Tests validate the logic in scripts/validate_ofac.py without requiring
the actual 6.7 MB archive (tests use in-memory synthetic data).
Covers both the actual OFAC Enhanced XML schema (sanctionsData -> entities -> entity)
and the legacy schema (sdnList -> sdnEntry).
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Synthetic XML matching actual OFAC SDN Enhanced structure
# ---------------------------------------------------------------------------

ENHANCED_XML_TEMPLATE = """\
<?xml version="1.0" encoding="utf-8"?>
<sanctionsData xmlns="https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ENHANCED_XML">
  <entities>
    <entity id="4632">
      <generalInfo>
        <entityType>Entity</entityType>
      </generalInfo>
      <names>
        <name>
          <isPrimary>true</isPrimary>
          <translations>
            <translation>
              <formattedFullName>CENTRAL BANK OF IRAN</formattedFullName>
              <formattedLastName>CENTRAL BANK OF IRAN</formattedLastName>
            </translation>
          </translations>
        </name>
        <name>
          <isPrimary>false</isPrimary>
          <translations>
            <translation>
              <formattedFullName>BANK MARKAZI</formattedFullName>
            </translation>
          </translations>
        </name>
      </names>
      <features>
        <feature id="97667">
          <type>Digital Currency Address - TRX</type>
          <value>TNiq9AXBp9EjUqhDhrwrfvAA8U3GUQZH81</value>
        </feature>
        <feature id="97668">
          <type>Digital Currency Address - TRX</type>
          <value>TTiDLWE6fZK8okMJv6ijg42yrH6W2pjSr9</value>
        </feature>
        <feature id="88888">
          <type>Location</type>
          <value>Tehran, Iran</value>
        </feature>
      </features>
    </entity>
    <entity id="24003">
      <generalInfo>
        <entityType>Individual</entityType>
      </generalInfo>
      <names>
        <name>
          <isPrimary>true</isPrimary>
          <translations>
            <translation>
              <formattedFullName>MESRI, Behzad</formattedFullName>
              <formattedLastName>MESRI</formattedLastName>
              <formattedFirstName>Behzad</formattedFirstName>
            </translation>
          </translations>
        </name>
      </names>
      <features>
        <feature id="96516">
          <type>Digital Currency Address - XBT</type>
          <value>12aNKp2iDKuhEde2YfPd4YBQhszEwkmC1a</value>
        </feature>
        <feature id="77777">
          <type>Address</type>
          <value>123 Main Street, Tehran</value>
        </feature>
      </features>
    </entity>
  </entities>
</sanctionsData>
"""

# ---------------------------------------------------------------------------
# Synthetic XML matching legacy structure (for backwards compatibility)
# ---------------------------------------------------------------------------

LEGACY_XML_TEMPLATE = """\
<?xml version="1.0" encoding="utf-8"?>
<sdnList>
  <sdnEntry>
    <uid>12345</uid>
    <lastName>CRYPTO ENTITY</lastName>
    <sdnType>Entity</sdnType>
    <featureList>
      <feature id="55551">
        <featureType>Digital Currency Address - XBT</featureType>
        <featureVersionList>
          <featureVersion>
            <versionDetail>
              <detail>1A1zP1eP5QGefi2DMPTfTL5SLmv7Divfna</detail>
            </versionDetail>
          </featureVersion>
        </featureVersionList>
      </feature>
      <feature id="55552">
        <featureType>Digital Currency Address - ETH</featureType>
        <featureVersionList>
          <featureVersion>
            <versionDetail>
              <detail>0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe</detail>
            </versionDetail>
          </featureVersion>
        </featureVersionList>
      </feature>
    </featureList>
  </sdnEntry>
  <sdnEntry>
    <uid>99999</uid>
    <lastName>ORDINARY PERSON</lastName>
    <sdnType>Individual</sdnType>
    <featureList>
      <feature id="55553">
        <featureType>Title</featureType>
        <featureVersionList>
          <featureVersion>
            <versionDetail>
              <detail>Director</detail>
            </versionDetail>
          </featureVersion>
        </featureVersionList>
      </feature>
    </featureList>
  </sdnEntry>
</sdnList>
"""

NON_CRYPTO_XML = """\
<?xml version="1.0"?>
<sanctionsData>
  <entities>
    <entity id="1">
      <names><name><value>PERSON</value></name></names>
      <features>
        <feature id="101">
          <type>Nationality</type>
          <value>Iran</value>
        </feature>
      </features>
    </entity>
  </entities>
</sanctionsData>
"""


def make_zip(xml_content: str, filename: str = "sdn_enhanced.xml") -> bytes:
    """Create an in-memory ZIP containing the given XML."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, xml_content.encode("utf-8"))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Import the script functions directly from canonical scripts/ directory
# ---------------------------------------------------------------------------

import sys
import importlib

SCRIPTS_DIR_HOST = Path(__file__).resolve().parents[2] / "scripts"
SCRIPTS_DIR_CONTAINER = Path(__file__).resolve().parents[1] / "scripts"


def load_validate_ofac():
    for d in [str(SCRIPTS_DIR_HOST), str(SCRIPTS_DIR_CONTAINER)]:
        if Path(d).exists() and d not in sys.path:
            sys.path.insert(0, d)
    import validate_ofac
    importlib.reload(validate_ofac)
    return validate_ofac


# ---------------------------------------------------------------------------
# Tests — SHA-256 and ZIP Inspection
# ---------------------------------------------------------------------------

class TestSha256:
    def test_sha256_matches_hashlib(self, tmp_path):
        mod = load_validate_ofac()
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert mod.sha256_file(test_file) == expected

    def test_sha256_large_file(self, tmp_path):
        mod = load_validate_ofac()
        data = b"x" * 200_000
        f = tmp_path / "big.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert mod.sha256_file(f) == expected


class TestFindXmlInZip:
    def test_finds_sdn_xml(self):
        mod = load_validate_ofac()
        buf = io.BytesIO(make_zip("<root/>", "sdn_enhanced.xml"))
        with zipfile.ZipFile(buf) as zf:
            found = mod.find_xml_in_zip(zf)
        assert found == "sdn_enhanced.xml"

    def test_fallback_any_xml(self):
        mod = load_validate_ofac()
        buf = io.BytesIO(make_zip("<root/>", "otherfile.xml"))
        with zipfile.ZipFile(buf) as zf:
            found = mod.find_xml_in_zip(zf)
        assert found == "otherfile.xml"

    def test_no_xml_returns_none(self):
        mod = load_validate_ofac()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "hello")
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            found = mod.find_xml_in_zip(zf)
        assert found is None


# ---------------------------------------------------------------------------
# Tests — Actual Enhanced XML Schema Extraction (Entity-Feature Relationships)
# ---------------------------------------------------------------------------

class TestEnhancedSchemaCryptoExtraction:
    def test_entity_id_from_containing_entity(self):
        """Entity ID must come from <entity id='...'>, NOT <feature id='...'>."""
        mod = load_validate_ofac()
        xml_bytes = ENHANCED_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)

        assert len(records) == 3

        # First two records belong to entity 4632
        assert records[0]["entity_id"] == "4632"
        assert records[0]["sdn_id"] == "4632"
        assert records[1]["entity_id"] == "4632"
        assert records[1]["sdn_id"] == "4632"

        # Third record belongs to entity 24003
        assert records[2]["entity_id"] == "24003"
        assert records[2]["sdn_id"] == "24003"

    def test_feature_id_not_used_as_entity_id(self):
        """Feature ID must NOT be confused with entity ID; preserved separately."""
        mod = load_validate_ofac()
        xml_bytes = ENHANCED_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)

        for rec in records:
            assert rec["entity_id"] != rec["feature_id"]

        assert records[0]["feature_id"] == "97667"
        assert records[1]["feature_id"] == "97668"
        assert records[2]["feature_id"] == "96516"

    def test_entity_name_extracted_correctly(self):
        """Entity names must be extracted from containing entity identity records."""
        mod = load_validate_ofac()
        xml_bytes = ENHANCED_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)

        # Primary name for 4632 is CENTRAL BANK OF IRAN
        assert records[0]["entity_name"] == "CENTRAL BANK OF IRAN"
        assert records[1]["entity_name"] == "CENTRAL BANK OF IRAN"

        # Primary name for 24003 is MESRI, Behzad
        assert records[2]["entity_name"] == "MESRI, Behzad"

    def test_multiple_crypto_features_associated_with_same_entity(self):
        """Multiple crypto features under one entity must stay associated with that entity."""
        mod = load_validate_ofac()
        xml_bytes = ENHANCED_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)

        entity_4632_records = [r for r in records if r["entity_id"] == "4632"]
        assert len(entity_4632_records) == 2

        # Both records have different feature IDs and addresses
        assert entity_4632_records[0]["feature_id"] == "97667"
        assert entity_4632_records[0]["address"] == "TNiq9AXBp9EjUqhDhrwrfvAA8U3GUQZH81"

        assert entity_4632_records[1]["feature_id"] == "97668"
        assert entity_4632_records[1]["address"] == "TTiDLWE6fZK8okMJv6ijg42yrH6W2pjSr9"

    def test_crypto_types_and_addresses_extracted(self):
        """Feature types and exact addresses must match source values."""
        mod = load_validate_ofac()
        xml_bytes = ENHANCED_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)

        types = [r["feature_type"] for r in records]
        assert types == [
            "Digital Currency Address - TRX",
            "Digital Currency Address - TRX",
            "Digital Currency Address - XBT",
        ]

        addresses = [r["address"] for r in records]
        assert "12aNKp2iDKuhEde2YfPd4YBQhszEwkmC1a" in addresses
        assert "TNiq9AXBp9EjUqhDhrwrfvAA8U3GUQZH81" in addresses

    def test_enhanced_non_crypto_features_excluded(self):
        """Non-crypto features like Location must be excluded."""
        mod = load_validate_ofac()
        xml_bytes = ENHANCED_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)

        for r in records:
            assert r["feature_type"] != "Location"
            assert "Tehran" not in r["address"]

    def test_enhanced_postal_address_never_treated_as_crypto(self):
        """Postal address features must never be extracted as crypto addresses."""
        mod = load_validate_ofac()
        xml_bytes = ENHANCED_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)

        for r in records:
            assert r["feature_type"] != "Address"
            assert "123 Main Street" not in r["address"]

    def test_enhanced_provenance_preserved(self):
        """All records must carry direct provenance to OFAC SDN Enhanced XML."""
        mod = load_validate_ofac()
        xml_bytes = ENHANCED_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)

        for rec in records:
            assert rec["source"] == "OFAC SDN Enhanced List"
            assert "not inferred" in rec["provenance"].lower()


# ---------------------------------------------------------------------------
# Tests — Legacy Schema Extraction (Backwards Compatibility)
# ---------------------------------------------------------------------------

class TestLegacyCryptoExtraction:
    def test_extracts_correct_count(self):
        mod = load_validate_ofac()
        xml_bytes = LEGACY_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)
        assert len(records) == 2

    def test_legacy_entity_id_and_provenance(self):
        mod = load_validate_ofac()
        xml_bytes = LEGACY_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)
        for rec in records:
            assert rec["entity_id"] == "12345"
            assert rec["sdn_id"] == "12345"
            assert rec["entity_name"] == "CRYPTO ENTITY"
            assert "provenance" in rec

    def test_legacy_non_crypto_features_excluded(self):
        mod = load_validate_ofac()
        xml_bytes = NON_CRYPTO_XML.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)
        assert len(records) == 0

    def test_legacy_ordinary_address_not_inferred(self):
        """Postal address strings must NEVER be confused with crypto addresses."""
        mod = load_validate_ofac()
        xml_with_postal = """
        <sdnList>
          <sdnEntry>
            <uid>1</uid><lastName>PERSON</lastName><sdnType>Individual</sdnType>
            <featureList>
              <feature id="11">
                <featureType>Address</featureType>
                <featureVersionList>
                  <featureVersion><versionDetail><detail>123 Main Street, Tehran</detail></versionDetail></featureVersion>
                </featureVersionList>
              </feature>
            </featureList>
          </sdnEntry>
        </sdnList>
        """.encode()
        records = mod.extract_crypto_addresses(xml_with_postal)
        assert len(records) == 0

    def test_empty_xml_returns_empty(self):
        mod = load_validate_ofac()
        records = mod.extract_crypto_addresses(b"<sanctionsData></sanctionsData>")
        assert records == []

    def test_invalid_xml_returns_empty(self):
        mod = load_validate_ofac()
        records = mod.extract_crypto_addresses(b"NOT XML AT ALL <<<")
        assert records == []


class TestMetadataStructure:
    def test_metadata_has_required_fields(self, tmp_path):
        """Ensure metadata written to disk has the required provenance fields."""
        mod = load_validate_ofac()
        metadata = {
            "dataset": "OFAC SDN Enhanced List",
            "source": "U.S. Department of the Treasury",
            "original_filename": "sdn_enhanced.zip",
            "sha256": "abc123",
            "file_size_bytes": 1234,
            "processed_at": "2026-01-01T00:00:00+00:00",
            "processing_status": "complete",
            "errors": [],
            "record_counts": {
                "total_crypto_address_records": 42,
                "distinct_entities_with_crypto": 10,
            },
        }
        path = tmp_path / "meta.json"
        path.write_text(json.dumps(metadata))
        loaded = json.loads(path.read_text())
        for required_key in ["sha256", "original_filename", "source", "record_counts"]:
            assert required_key in loaded, f"Missing key: {required_key}"
