from typing import List
from app.attribution.models import VASPAddressRecord
from app.attribution.repository import VASPAttributionRepository

class VASPAttributionRegistry:
    """
    Deterministic, provenance-aware VASP/Exchange attribution registry.
    Acts as a read-only evidence source answering "Do we have an attribution record for this wallet address on this blockchain".
    """
    def __init__(self, repository: VASPAttributionRepository):
        self.repository = repository
        
    async def lookup(self, address: str, chain: str) -> List[VASPAddressRecord]:
        """
        Exact address lookup. Returns multiple candidates if evidence conflicts.
        Handles unknown addresses cleanly (returns empty list).
        Isolates chains (same address on different chains are distinct).
        """
        if not address or not chain:
            return []
        return await self.repository.lookup(address=address, chain=chain)
