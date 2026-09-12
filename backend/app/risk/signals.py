"""CHAKRA Step 5: Deterministic Signal Adapters."""
from __future__ import annotations
import logging
from typing import List, Optional

from app.schemas.risk_scoring import DeterministicSignals, SignalState
from app.schemas.alert import SanctionedAddressHit
from app.forensics.typology_detector import TypologyDetection, TypologyType
from app.forensics.risk_engine import RiskEngine

logger = logging.getLogger(__name__)

class SignalAdapter:
    """Adapts upstream CHAKRA deterministic signals for risk fusion."""

    @staticmethod
    def evaluate_signals(
        target_address: str,
        chain: str,
        network: str,
        sanctions_hits: List[SanctionedAddressHit],
        typologies: List[TypologyDetection]
    ) -> DeterministicSignals:
        """Evaluate deterministic signals without fabricating unavailable states."""
        
        # 1. Rule Engine Signal
        # We use the existing deterministic RiskEngine to get a base rule evaluation.
        # If the base engine yields a score > 0, we consider the rule engine signal TRUE.
        try:
            base_record = RiskEngine.evaluate(
                target_address=target_address,
                chain=chain,
                network=network,
                sanctions_hits=sanctions_hits,
                typologies=typologies
            )
            rule_state = "TRUE" if base_record.overall_score > 0 else "FALSE"
        except Exception as e:
            logger.warning(f"Rule engine evaluation failed: {e}")
            rule_state = "UNKNOWN"

        # 2. Known Bad Address Hit (Sanctions)
        valid_sanctions = [h for h in sanctions_hits if not RiskEngine._is_synthetic(h.source, h.evidence_id)]
        if valid_sanctions:
            bad_address_state = SignalState(state="TRUE", evidence_id=valid_sanctions[0].evidence_id)
        else:
            bad_address_state = SignalState(state="FALSE")

        # 3. Mixer Signal
        # TypologyDetector currently returns "insufficient_evidence" for mixers to be safe.
        mixer_typos = [t for t in typologies if t.typology_type == TypologyType.MIXER_INTERACTION]
        if mixer_typos:
            t = mixer_typos[0]
            if t.confidence_level == "observed":
                mixer_state = SignalState(state="TRUE", evidence_id=t.detection_id)
            else:
                # If it explicitly returned insufficient_evidence, we mark it UNAVAILABLE/UNKNOWN, not FALSE.
                mixer_state = SignalState(state="UNKNOWN")
        else:
            mixer_state = SignalState(state="UNKNOWN") # If not even checked or provided

        # 4. Cross Chain Signal
        bridge_typos = [t for t in typologies if t.typology_type == TypologyType.CROSS_CHAIN]
        if bridge_typos:
            t = bridge_typos[0]
            if t.confidence_level == "observed":
                bridge_state = SignalState(state="TRUE", evidence_id=t.detection_id)
            else:
                bridge_state = SignalState(state="UNKNOWN")
        else:
            bridge_state = SignalState(state="UNKNOWN")

        # 5. Temporal Signal (Rapid Hopping)
        rapid_typos = [t for t in typologies if t.typology_type == TypologyType.RAPID_HOPPING]
        if rapid_typos:
            t = rapid_typos[0]
            if t.confidence_level == "observed":
                temporal_state = SignalState(state="TRUE", evidence_id=t.detection_id)
            else:
                temporal_state = SignalState(state="FALSE")
        else:
            temporal_state = SignalState(state="FALSE")

        return DeterministicSignals(
            rule_engine_signal=SignalState(state=rule_state),
            known_bad_address_hit=bad_address_state,
            mixer_signal=mixer_state,
            cross_chain_signal=bridge_state,
            temporal_signal=temporal_state
        )
