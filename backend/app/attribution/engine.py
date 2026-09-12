from typing import List, Optional, Dict, Tuple
from app.schemas.chain import Chain
from app.attribution.models import VASPAddressRecord
from app.attribution.registry import VASPAttributionRegistry
from app.attribution.authority import AuthorityPolicy, AuthorityClass
from app.attribution.engine_models import AttributionDecision, AttributionConfidence, AttributionProvenance
from app.clustering.models import AddressCluster

class AttributionDecisionEngine:
    """
    Step 4.2 Attribution Decision Engine.
    Determines categorical attribution confidence from Step 4.1 registry observations and Step 3 deterministic evidence.
    """
    def __init__(self, registry: VASPAttributionRegistry):
        self.registry = registry

    def _determine_candidate_confidence(
        self,
        vasp_name: str,
        direct_obs: List[VASPAddressRecord],
        inferred_obs: List[VASPAddressRecord],
        cluster: Optional[AddressCluster]
    ) -> Tuple[AttributionConfidence, str]:
        """
        Evaluate a single candidate vasp_name to determine its confidence tier.
        """
        # If we have direct observations on the target address
        if direct_obs:
            # Check for authoritative source via explicit AuthorityPolicy contract
            for obs in direct_obs:
                if AuthorityPolicy.is_authoritative(obs):
                    return AttributionConfidence.CONFIRMED, "Direct VASP registry attribution supported by authoritative source evidence."
            
            # Strong corroboration requires either:
            # 1. Multiple independent sources directly on this address (different provenance sources)
            # 2. Consistent deterministic cluster corroboration (a cluster sibling also has an attribution to this same VASP)
            unique_sources = {obs.source for obs in direct_obs}
            has_multiple_sources = len(unique_sources) > 1
            has_cluster_corroboration = len(inferred_obs) > 0
            
            if has_multiple_sources and has_cluster_corroboration:
                return AttributionConfidence.HIGH_CONFIDENCE, "VASP attribution supported by multiple independent direct observations and corroborating cluster sibling attribution."
            elif has_cluster_corroboration:
                return AttributionConfidence.HIGH_CONFIDENCE, "VASP attribution supported by direct observation and corroborating attribution on cluster sibling(s)."
            elif has_multiple_sources:
                return AttributionConfidence.HIGH_CONFIDENCE, "VASP attribution supported by multiple independent direct registry observations."
            
            # Direct observation without corroboration or authoritative source
            return AttributionConfidence.PROBABLE_INFERRED, "Direct VASP registry observation without authoritative source or independent corroboration; classified under PROBABLE_INFERRED tier per four-tier design constraint."
            
        # If we have no direct observations, but we have inferred observations from a cluster sibling
        if inferred_obs:
            return AttributionConfidence.PROBABLE_INFERRED, "VASP attribution derived from deterministic Step 3 clustering inference from sibling member(s), without direct confirmation."
            
        return AttributionConfidence.UNKNOWN, "No sufficient attribution evidence was available."

    async def evaluate(self, address: str, chain: Chain, cluster: Optional[AddressCluster] = None) -> AttributionDecision:
        """
        Evaluate attribution for a given address.
        Optionally takes a Step 3 AddressCluster to support deterministic inference and corroboration.
        """
        # 1. Fetch direct observations for the target address
        direct_observations = await self.registry.lookup(address, chain.value)
        
        # 2. Fetch inferred observations from cluster siblings (if any)
        inferred_observations = []
        if cluster:
            for member in cluster.members:
                if member.normalized_address != address:
                    sibling_obs = await self.registry.lookup(member.normalized_address, chain.value)
                    inferred_observations.extend(sibling_obs)
                    
        all_observations = direct_observations + inferred_observations
        
        # 3. Handle base case: No evidence whatsoever
        if not all_observations:
            return AttributionDecision(
                address=address,
                chain=chain,
                confidence=AttributionConfidence.UNKNOWN,
                vasp_name=None,
                service_type=None,
                candidate_attributions=[],
                provenance=AttributionProvenance(authority_class=AuthorityClass.UNKNOWN),
                explanation="No sufficient attribution evidence was available."
            )
            
        # 4. Group observations by VASP name
        candidates: Dict[str, Dict[str, List[VASPAddressRecord]]] = {}
        for obs in direct_observations:
            candidates.setdefault(obs.vasp_name, {"direct": [], "inferred": []})["direct"].append(obs)
        for obs in inferred_observations:
            candidates.setdefault(obs.vasp_name, {"direct": [], "inferred": []})["inferred"].append(obs)
            
        # 5. Evaluate confidence for each candidate
        tier_scores = {
            AttributionConfidence.CONFIRMED: 4,
            AttributionConfidence.HIGH_CONFIDENCE: 3,
            AttributionConfidence.PROBABLE_INFERRED: 2,
            AttributionConfidence.UNKNOWN: 1
        }
        
        best_tier = AttributionConfidence.UNKNOWN
        best_score = 0
        best_candidates = []
        candidate_explanations = {}
        
        for vasp_name in sorted(candidates.keys()):
            obs_groups = candidates[vasp_name]
            conf, expl = self._determine_candidate_confidence(
                vasp_name, obs_groups["direct"], obs_groups["inferred"], cluster
            )
            score = tier_scores[conf]
            candidate_explanations[vasp_name] = expl
            
            if score > best_score:
                best_score = score
                best_tier = conf
                best_candidates = [vasp_name]
            elif score == best_score:
                best_candidates.append(vasp_name)
                
        # 6. Resolve final decision
        # If multiple candidates have equivalent strong evidence, we do not fabricate certainty.
        if len(best_candidates) == 1 and best_tier != AttributionConfidence.UNKNOWN:
            winner = best_candidates[0]
            # Grab service_type from the first observation of the winner
            winner_obs = candidates[winner]["direct"] + candidates[winner]["inferred"]
            service_type = winner_obs[0].service_type if winner_obs else None
            corroborating_count = max(0, len(candidates[winner]["direct"]) + len(candidates[winner]["inferred"]) - 1)
            primary_obs = candidates[winner]["direct"][0] if candidates[winner]["direct"] else candidates[winner]["inferred"][0]
            winner_authority = AuthorityPolicy.classify_observation(primary_obs)
            
            return AttributionDecision(
                address=address,
                chain=chain,
                confidence=best_tier,
                vasp_name=winner,
                service_type=service_type,
                candidate_attributions=all_observations,
                provenance=AttributionProvenance(
                    direct_observations=candidates[winner]["direct"],
                    inferred_observations=candidates[winner]["inferred"],
                    cluster_id=cluster.cluster_id if cluster else None,
                    corroborating_evidence_count=corroborating_count,
                    authority_class=winner_authority
                ),
                explanation=candidate_explanations[winner]
            )
        else:
            # Ambiguity or UNKNOWN
            return AttributionDecision(
                address=address,
                chain=chain,
                confidence=AttributionConfidence.UNKNOWN,
                vasp_name=None,
                service_type=None,
                candidate_attributions=all_observations,
                provenance=AttributionProvenance(
                    direct_observations=direct_observations,
                    inferred_observations=inferred_observations,
                    cluster_id=cluster.cluster_id if cluster else None,
                    corroborating_evidence_count=0,
                    authority_class=AuthorityClass.UNKNOWN
                ),
                explanation="Multiple attribution candidates have equivalent strong evidence; cannot resolve deterministically." if best_candidates else "No sufficient attribution evidence was available."
            )
