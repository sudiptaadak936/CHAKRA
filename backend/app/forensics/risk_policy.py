"""Risk scoring policy and deterministic constants for CHAKRA."""

# Risk Level Thresholds
# These represent conceptual bands for investigational prioritization.
CRITICAL_THRESHOLD = 80.0
HIGH_THRESHOLD = 60.0
MEDIUM_THRESHOLD = 35.0
LOW_THRESHOLD = 10.0

# Score Bounds
MAX_SCORE = 100.0
MIN_SCORE = 0.0

# Component Weights (Engineering choices, not probabilities)
# Sanctions
SANCTIONS_HIT = 100.0

# Typologies (Only eligible if confidence == "observed")
TYPOLOGY_WEIGHTS = {
    "PEEL_CHAIN": 25.0,
    "RAPID_HOPPING": 20.0,
    "FAN_IN": 15.0,
    "FAN_OUT": 15.0,
}

# The following explicitly contribute 0.0 points
INELIGIBLE_TYPOLOGIES = [
    "MIXER_INTERACTION",
    "CROSS_CHAIN",
]

