"""CHAKRA Step 6: Suspect Registry Tests."""
import pytest
from app.forensics.fingerprint import GraphFingerprint
from app.attribution.suspect_registry import SuspectRegistryService, SuspectRegistryMatchResult

def test_fingerprint_generation():
    """Verify deterministic 6D bounded fingerprint generation."""
    vec1 = GraphFingerprint.generate(
        degree=10, transaction_volume=5000, cluster_size=5,
        amount_retention_ratio=0.5, inter_hop_velocity=1.2, days_active=10
    )
    
    assert len(vec1) == 6
    # Check L2 normalization
    import math
    norm = math.sqrt(sum(v*v for v in vec1))
    assert abs(norm - 1.0) < 1e-5
    
    # Check bounds internally (all should be positive after our mapping logic)
    assert all(v >= 0.0 for v in vec1)

@pytest.mark.asyncio
async def test_registry_lookup_no_db_fallback():
    """Verify registry fails gracefully when DB table missing/unavailable."""
    from unittest.mock import AsyncMock
    mock_pool = AsyncMock()
    mock_pool.acquire.side_effect = Exception("DB Down")
    
    service = SuspectRegistryService(pool=mock_pool)
    res = await service.lookup("addr1", "bitcoin")
    
    # IMPORTANT INVARIANT: MUST return UNKNOWN if it can't definitively check
    assert res.match_result == SuspectRegistryMatchResult.UNKNOWN

@pytest.mark.asyncio
async def test_registry_lookup_match():
    """Verify registry returns MATCH and ID when found."""
    from unittest.mock import MagicMock, AsyncMock
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    mock_conn.fetchrow.return_value = {
        "registry_id": "123e4567-e89b-12d3-a456-426614174000",
        "risk_score": 95.0,
        "first_seen_case_id": "case-xyz"
    }
    
    service = SuspectRegistryService(pool=mock_pool)
    res = await service.lookup("addr1", "bitcoin")
    
    assert res.match_result == SuspectRegistryMatchResult.MATCH
    assert res.match_id == "123e4567-e89b-12d3-a456-426614174000"
    assert "95.0" in res.reason

@pytest.mark.asyncio
async def test_registry_lookup_no_match():
    """Verify registry returns NO_MATCH when table is queried but no row exists."""
    from unittest.mock import MagicMock, AsyncMock
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    mock_conn.fetchrow.return_value = None
    
    service = SuspectRegistryService(pool=mock_pool)
    res = await service.lookup("addr2", "bitcoin")
    
    # IMPORTANT INVARIANT: MUST return NO_MATCH (not UNKNOWN) since we checked
    assert res.match_result == SuspectRegistryMatchResult.NO_MATCH

@pytest.mark.asyncio
async def test_similar_case_search():
    """Verify similarity search parses results correctly."""
    from unittest.mock import MagicMock, AsyncMock
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    # Mock return rows
    mock_conn.fetch.return_value = [
        {
            "registry_id": "123",
            "address": "addr3",
            "chain": "bitcoin",
            "first_seen_case_id": "case1",
            "risk_score": 90.0,
            "similarity": 0.95
        }
    ]
    
    service = SuspectRegistryService(pool=mock_pool)
    vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    res = await service.find_similar_cases(vec, limit=1)
    
    assert len(res) == 1
    assert res[0]["similarity"] == 0.95
    assert res[0]["registry_id"] == "123"
    assert res[0]["address"] == "addr3"
