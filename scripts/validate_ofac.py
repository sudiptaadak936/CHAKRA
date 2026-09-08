#!/usr/bin/env python3
"""
CHAKRA Step 0.5 — OFAC SDN Enhanced Dataset Validation Script

Validates the OFAC Specially Designated Nationals (SDN) Enhanced XML dataset,
extracts cryptocurrency identifiers with true containing-entity relationships,
and writes processed output and provenance metadata.

Source: https://ofac.treasury.gov/specially-designated-nationals-list-data-formats-data-schemas
File:   data/ofac/raw/sdn_enhanced.zip

Usage:
    python scripts/validate_ofac.py

Outputs:
    data/ofac/processed/sdn_crypto_addresses.json   — extracted crypto identifiers
    data/ofac/metadata/ofac_metadata.json           — provenance and statistics

IMPORTANT:
    The raw ZIP file is never modified.
    Each record associates the digital currency feature with its containing <entity>.
    The containing entity ID is preserved as `entity_id` (and aliased as `sdn_id`).
    The individual cryptocurrency feature ID is preserved as `feature_id`.
    Entity names are extracted from the containing entity's identity elements.
    Crypto addresses are identified from explicit 'Digital Currency Address'
    feature records in the SDN data — never inferred from names or postal addresses.
"""

import hashlib
import json
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "ofac" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "ofac" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "ofac" / "metadata"
ZIP_PATH = RAW_DIR / "sdn_enhanced.zip"

# Feature type names that indicate a digital currency address in the SDN Enhanced schema
CRYPTO_FEATURE_TYPES = {
    "Digital Currency Address - XBT",   # Bitcoin
    "Digital Currency Address - ETH",   # Ethereum
    "Digital Currency Address - XMR",   # Monero
    "Digital Currency Address - LTC",   # Litecoin
    "Digital Currency Address - ZEC",   # Zcash
    "Digital Currency Address - DASH",  # Dash
    "Digital Currency Address - BTG",   # Bitcoin Gold
    "Digital Currency Address - ETC",   # Ethereum Classic
    "Digital Currency Address - BSV",   # Bitcoin SV
    "Digital Currency Address - BCH",   # Bitcoin Cash
    "Digital Currency Address - XRP",   # XRP/Ripple
    "Digital Currency Address - XLM",   # Stellar
    "Digital Currency Address - TRX",   # TRON
    "Digital Currency Address - USDT",  # Tether (ERC-20 / TRC-20)
    "Digital Currency Address - USDC",  # USD Coin
    "Digital Currency Address - BUSD",  # Binance USD
    "Digital Currency Address - ARB",   # Arbitrum
    "Digital Currency Address - MATIC", # Polygon/Matic
    "Digital Currency Address",         # Generic fallback
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_xml_in_zip(zf: zipfile.ZipFile) -> str | None:
    """Return the name of the primary SDN XML file inside the archive."""
    for name in zf.namelist():
        lower = name.lower()
        if lower.endswith(".xml") and ("sdn" in lower or "enhanced" in lower):
            return name
    # Fallback: any .xml file
    for name in zf.namelist():
        if name.lower().endswith(".xml"):
            return name
    return None


def strip_ns(tag: str) -> str:
    """Strip XML namespace prefix from a tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def is_crypto_feature_type(ftype: str | None) -> bool:
    """Determine whether a feature type represents a cryptocurrency address."""
    if not ftype:
        return False
    ft = ftype.strip()
    return ft in CRYPTO_FEATURE_TYPES or ft.startswith("Digital Currency Address")


def extract_crypto_addresses(xml_bytes: bytes) -> list[dict]:
    """Parse the SDN Enhanced XML and extract digital currency address records.

    Handles:
    1. The actual OFAC Enhanced XML schema (root: sanctionsData), where entities
       are structured as:
         sanctionsData -> entities -> entity (id=...)
           -> generalInfo -> entityType
           -> names -> name -> isPrimary, formattedFullName/formattedLastName
           -> features -> feature (id=...) -> type, value
       Extracts entity_id from the containing entity, feature_id from the feature,
       and the primary or fallback formatted name.

    2. The legacy sdnList/sdnEntry schema for backward-compatibility with older tests.

    Returns a list of dicts with provenance fields. Never infers addresses
    from names or postal address fields.
    """
    records = []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        print(f"  [ERROR] XML parse error: {exc}", file=sys.stderr)
        return records

    # -----------------------------------------------------------------------
    # Strategy 1: OFAC Enhanced XML Schema (<entity> elements)
    # -----------------------------------------------------------------------
    for entity in root.iter():
        if strip_ns(entity.tag) != "entity":
            continue

        entity_id = entity.get("id")
        entity_name = None
        entity_type = None

        primary_name = None
        fallback_name = None

        names_elem = None
        features_elem = None

        for child in entity:
            ctag = strip_ns(child.tag)
            if ctag == "names":
                names_elem = child
            elif ctag == "features":
                features_elem = child
            elif ctag == "generalInfo":
                for gc in child:
                    if strip_ns(gc.tag) == "entityType":
                        entity_type = gc.text.strip() if gc.text else None

        if names_elem is not None:
            for name_entry in names_elem:
                is_primary = False
                entry_name = None
                for nc in name_entry:
                    if strip_ns(nc.tag) == "isPrimary" and nc.text and nc.text.strip().lower() == "true":
                        is_primary = True
                for nc in name_entry.iter():
                    nctag = strip_ns(nc.tag)
                    if nctag == "formattedFullName" and nc.text and nc.text.strip():
                        entry_name = nc.text.strip()
                        break
                    elif nctag in ("formattedLastName", "value") and nc.text and nc.text.strip() and not entry_name:
                        entry_name = nc.text.strip()
                if entry_name:
                    if is_primary and not primary_name:
                        primary_name = entry_name
                    elif not fallback_name:
                        fallback_name = entry_name

        # Fallback if no names element
        if not primary_name and not fallback_name:
            for elem in entity.iter():
                etag = strip_ns(elem.tag)
                if etag in ("name", "lastName", "primaryName") and elem.text and elem.text.strip():
                    fallback_name = elem.text.strip()
                    break

        entity_name = primary_name or fallback_name or None

        # Process features within this containing entity
        feature_containers = [features_elem] if features_elem is not None else [entity]
        for container in feature_containers:
            for feat in container.iter():
                if strip_ns(feat.tag) != "feature":
                    continue
                feat_id = feat.get("id")
                ftype = None
                fval = None
                for fc in feat:
                    fctag = strip_ns(fc.tag)
                    if fctag == "type":
                        ftype = fc.text.strip() if fc.text else None
                    elif fctag == "value":
                        fval = fc.text.strip() if fc.text else None

                if ftype and is_crypto_feature_type(ftype) and fval:
                    records.append({
                        "entity_id": str(entity_id) if entity_id is not None else None,
                        "sdn_id": str(entity_id) if entity_id is not None else None,  # backward compatibility
                        "feature_id": str(feat_id) if feat_id is not None else None,
                        "entity_name": entity_name,
                        "entity_type": entity_type,
                        "feature_type": ftype,
                        "address": fval,
                        "source": "OFAC SDN Enhanced List",
                        "provenance": "Directly extracted from OFAC SDN Enhanced XML; not inferred.",
                    })

    # -----------------------------------------------------------------------
    # Strategy 2: Legacy sdnList/sdnEntry Schema (fallback)
    # -----------------------------------------------------------------------
    if not records:
        for entry in root.iter():
            if strip_ns(entry.tag) != "sdnEntry":
                continue

            sdn_id = None
            entity_name = None
            sdn_type = None

            for child in entry:
                ctag = strip_ns(child.tag)
                if ctag == "uid":
                    sdn_id = child.text.strip() if child.text else None
                elif ctag == "lastName":
                    entity_name = child.text.strip() if child.text else None
                elif ctag == "sdnType":
                    sdn_type = child.text.strip() if child.text else None

            for feat in entry.iter():
                if strip_ns(feat.tag) != "feature":
                    continue

                feat_id = feat.get("id")
                ftype = None
                for fc in feat:
                    if strip_ns(fc.tag) == "featureType":
                        ftype = fc.text.strip() if fc.text else None
                        break

                if not ftype or not is_crypto_feature_type(ftype):
                    continue

                for ver in feat.iter():
                    if strip_ns(ver.tag) == "detail" and ver.text and ver.text.strip():
                        records.append({
                            "entity_id": str(sdn_id) if sdn_id is not None else None,
                            "sdn_id": str(sdn_id) if sdn_id is not None else None,
                            "feature_id": str(feat_id) if feat_id is not None else None,
                            "entity_name": entity_name,
                            "entity_type": sdn_type,
                            "feature_type": ftype,
                            "address": ver.text.strip(),
                            "source": "OFAC SDN Enhanced List",
                            "provenance": "Directly extracted from OFAC SDN Enhanced XML; not inferred.",
                        })

    return records


def main() -> int:
    print("=" * 60)
    print("  CHAKRA — OFAC SDN Enhanced Dataset Validation")
    print("=" * 60)

    # Ensure output directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    errors = []

    # ------------------------------------------------------------------
    # Step 1 — Verify ZIP is readable
    # ------------------------------------------------------------------
    print(f"\n[1] Verifying archive: {ZIP_PATH.name}")
    if not ZIP_PATH.exists():
        print(f"  [FAIL] File not found: {ZIP_PATH}")
        return 1

    file_size_bytes = ZIP_PATH.stat().st_size
    print(f"  File size: {file_size_bytes:,} bytes ({file_size_bytes / 1024 / 1024:.2f} MB)")

    if not zipfile.is_zipfile(ZIP_PATH):
        print("  [FAIL] File is not a valid ZIP archive.")
        return 1
    print("  ZIP integrity: OK")

    # ------------------------------------------------------------------
    # Step 2 — Calculate SHA-256
    # ------------------------------------------------------------------
    print("\n[2] Computing SHA-256...")
    sha256 = sha256_file(ZIP_PATH)
    print(f"  SHA-256: {sha256}")

    # ------------------------------------------------------------------
    # Step 3 — Inspect archive contents
    # ------------------------------------------------------------------
    print("\n[3] Inspecting archive contents...")
    archive_contents = []
    xml_filename = None
    xml_bytes = None

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for info in zf.infolist():
            archive_contents.append({
                "filename": info.filename,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
            })
            print(f"  {info.filename:50s}  {info.file_size:>12,} bytes uncompressed")

        xml_filename = find_xml_in_zip(zf)
        if xml_filename:
            print(f"\n  Primary XML identified: {xml_filename}")
            with zf.open(xml_filename) as xf:
                xml_bytes = xf.read()
        else:
            print("  [WARN] No XML file found in archive. Checking for other formats...")
            errors.append("No XML file found in archive.")

    # ------------------------------------------------------------------
    # Step 4 — Extract crypto address records
    # ------------------------------------------------------------------
    crypto_records = []
    if xml_bytes:
        print("\n[4] Extracting digital currency address records...")
        crypto_records = extract_crypto_addresses(xml_bytes)
        print(f"  Crypto address records found: {len(crypto_records)}")

        # Currency type breakdown
        type_counts: dict[str, int] = {}
        for rec in crypto_records:
            ft = rec["feature_type"]
            type_counts[ft] = type_counts.get(ft, 0) + 1
        for ft, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {ft}: {count}")
    else:
        print("\n[4] Skipping crypto extraction (no XML content available).")

    # ------------------------------------------------------------------
    # Step 5 — Write processed output
    # ------------------------------------------------------------------
    print("\n[5] Writing processed output...")
    processed_path = PROCESSED_DIR / "sdn_crypto_addresses.json"
    processed_data = {
        "description": (
            "Digital currency addresses extracted from the OFAC SDN Enhanced dataset. "
            "Each record maps a cryptocurrency feature (feature_id) to its containing "
            "SDN entity (entity_id, aliased as sdn_id for backward compatibility) and primary entity name. "
            "All records carry direct provenance to the source XML. "
            "Addresses are NEVER inferred from names or postal addresses."
        ),
        "source": "OFAC SDN Enhanced List",
        "source_url": "https://ofac.treasury.gov/specially-designated-nationals-list-data-formats-data-schemas",
        "source_file": ZIP_PATH.name,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(crypto_records),
        "records": crypto_records,
    }
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print(f"  Written: {processed_path.relative_to(PROJECT_ROOT)}")
    print(f"  Records: {len(crypto_records)}")

    # ------------------------------------------------------------------
    # Step 6 — Write metadata / provenance
    # ------------------------------------------------------------------
    print("\n[6] Writing metadata...")

    distinct_entity_ids = len({r["entity_id"] for r in crypto_records if r.get("entity_id")})

    # Currency type summary for metadata
    type_summary = {}
    for rec in crypto_records:
        ft = rec["feature_type"]
        type_summary[ft] = type_summary.get(ft, 0) + 1

    metadata = {
        "dataset": "OFAC SDN Enhanced List",
        "source": "U.S. Department of the Treasury, Office of Foreign Assets Control (OFAC)",
        "source_url": "https://ofac.treasury.gov/specially-designated-nationals-list-data-formats-data-schemas",
        "original_filename": ZIP_PATH.name,
        "acquisition_note": "Downloaded from public OFAC website. No account or API key required.",
        "sha256": sha256,
        "file_size_bytes": file_size_bytes,
        "archive_contents": archive_contents,
        "primary_xml_file": xml_filename,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "processing_status": "complete" if not errors else "partial",
        "errors": errors,
        "record_counts": {
            "total_crypto_address_records": len(crypto_records),
            "distinct_entities_with_crypto": distinct_entity_ids,
            "distinct_sdn_entries_with_crypto": distinct_entity_ids,
            "by_currency_type": type_summary,
        },
        "schema_notes": (
            "SDN Enhanced XML schema. Cryptocurrency addresses are stored as "
            "<feature id='...'> elements under containing <entity id='...'> records, "
            "with <type> starting with 'Digital Currency Address' and <value> containing the address."
        ),
        "license_note": (
            "OFAC SDN data is a U.S. government publication and is in the public domain. "
            "Re-use is permitted. Attribution to OFAC/Treasury is recommended."
        ),
    }

    metadata_path = METADATA_DIR / "ofac_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Written: {metadata_path.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if errors:
        print(f"  OFAC Validation: PARTIAL ({len(errors)} warning(s))")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  OFAC Validation: PASS")
    print("=" * 60)
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
