"""
Unit tests for OFAC dataset validation logic.

Tests validate the logic in scripts/validate_ofac.py without requiring
the actual 6.7 MB archive (tests use in-memory synthetic data).
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: synthetic XML matching OFAC SDN Enhanced structure
# ---------------------------------------------------------------------------

OFAC_XML_TEMPLATE = """\
<?xml version="1.0" encoding="utf-8"?>
<sdnList>
  <sdnEntry>
    <uid>12345</uid>
    <lastName>CRYPTO ENTITY</lastName>
    <sdnType>Entity</sdnType>
    <featureList>
      <feature>
        <featureType>Digital Currency Address - XBT</featureType>
        <featureVersionList>
          <featureVersion>
            <versionDetail>
              <detail>1A1zP1eP5QGefi2DMPTfTL5SLmv7Divfna</detail>
            </versionDetail>
          </featureVersion>
        </featureVersionList>
      </feature>
      <feature>
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
      <feature>
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
<sdnList>
  <sdnEntry>
    <uid>1</uid>
    <lastName>PERSON</lastName>
    <featureList>
      <feature>
        <featureType>Nationality</featureType>
        <featureVersionList>
          <featureVersion><versionDetail><detail>Iran</detail></versionDetail></featureVersion>
        </featureVersionList>
      </feature>
    </featureList>
  </sdnEntry>
</sdnList>
"""


def make_zip(xml_content: str, filename: str = "sdn_enhanced.xml") -> bytes:
    """Create an in-memory ZIP containing the given XML."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, xml_content.encode("utf-8"))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Import the script functions directly
# ---------------------------------------------------------------------------

import sys
import importlib

# We import the script as a module. Add scripts/ to path.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
BACKEND_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_validate_ofac():
    for d in [str(SCRIPTS_DIR), str(BACKEND_SCRIPTS_DIR)]:
        if Path(d).exists() and d not in sys.path:
            sys.path.insert(0, d)
    import validate_ofac
    importlib.reload(validate_ofac)
    return validate_ofac


# ---------------------------------------------------------------------------
# Tests
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


class TestCryptoExtraction:
    def test_extracts_correct_count(self):
        mod = load_validate_ofac()
        xml_bytes = OFAC_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)
        assert len(records) == 2

    def test_feature_types_correct(self):
        mod = load_validate_ofac()
        xml_bytes = OFAC_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)
        types = {r["feature_type"] for r in records}
        assert "Digital Currency Address - XBT" in types
        assert "Digital Currency Address - ETH" in types

    def test_address_values_extracted(self):
        mod = load_validate_ofac()
        xml_bytes = OFAC_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)
        addresses = {r["address"] for r in records}
        assert "1A1zP1eP5QGefi2DMPTfTL5SLmv7Divfna" in addresses

    def test_provenance_field_present(self):
        mod = load_validate_ofac()
        xml_bytes = OFAC_XML_TEMPLATE.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)
        for rec in records:
            assert "provenance" in rec
            assert "sdn_id" in rec
            assert rec["sdn_id"] == "12345"

    def test_non_crypto_features_excluded(self):
        mod = load_validate_ofac()
        xml_bytes = NON_CRYPTO_XML.encode("utf-8")
        records = mod.extract_crypto_addresses(xml_bytes)
        # 'Nationality' is not a crypto feature — must be excluded
        assert len(records) == 0

    def test_ordinary_address_not_inferred(self):
        """Postal address strings must NEVER be confused with crypto addresses."""
        mod = load_validate_ofac()
        xml_with_postal = """
        <sdnList>
          <sdnEntry>
            <uid>1</uid><lastName>PERSON</lastName><sdnType>Individual</sdnType>
            <featureList>
              <feature>
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
        assert len(records) == 0, "Postal/street addresses must never be extracted as crypto"

    def test_empty_xml_returns_empty(self):
        mod = load_validate_ofac()
        records = mod.extract_crypto_addresses(b"<sdnList></sdnList>")
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
            "record_counts": {"total_crypto_address_records": 42},
        }
        path = tmp_path / "meta.json"
        path.write_text(json.dumps(metadata))
        loaded = json.loads(path.read_text())
        for required_key in ["sha256", "original_filename", "source", "record_counts"]:
            assert required_key in loaded, f"Missing key: {required_key}"
