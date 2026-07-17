"""
Map eigenvector components to 12 phronesis modes.
"""

import numpy as np
from icm_tokenizer import Token


MODES = [
    "ground",
    "know",
    "see",
    "flow",
    "grow",
    "ignite",
    "learn",
    "connect",
    "transform",
    "integrate",
    "receive",
    "reflect",
]


class ModeRouter:
    """Route eigenvector components to phronesis modes."""

    # Map token groups to mode indices
    TOKEN_TO_MODE = {
        "Entity": 0,        # Ground (foundation, entity)
        "Action": 3,        # Flow (movement, navigation)
        "Property": 2,      # See (perception, qualities)
        "Goal": 4,          # Grow (development, becoming)
        "Context": 7,       # Connect (relationship, context)
    }

    def activate(self, tokens: list[Token], eigenvector: np.ndarray) -> dict:
        """
        Map eigenvector components to mode activation.

        Returns:
            {
                "mode_activation": [12 floats in 0-1],
                "dominant_mode": str,
                "mode_entropy": float,
            }
        """
        # Initialize mode energy
        mode_energy = np.zeros(len(MODES))

        # Accumulate eigenvector energy by token type
        for i, token in enumerate(tokens):
            if i < len(eigenvector) and not token.is_boundary:
                mode_idx = self.TOKEN_TO_MODE.get(token.type, 0)
                # Energy = absolute eigenvector component
                mode_energy[mode_idx] += abs(eigenvector[i])

        # Normalize to probability distribution
        if np.sum(mode_energy) > 0:
            mode_activation = mode_energy / np.sum(mode_energy)
        else:
            # Fallback: uniform distribution
            mode_activation = np.ones(len(MODES)) / len(MODES)

        # Dominant mode
        dominant_idx = np.argmax(mode_activation)
        dominant_mode = MODES[dominant_idx]

        # Entropy (measure of mode distribution spread)
        # Lower entropy = more coherent (energy concentrated in few modes)
        mode_entropy = float(-np.sum(mode_activation * np.log(mode_activation + 1e-10)))

        return {
            "mode_activation": mode_activation.tolist(),
            "dominant_mode": dominant_mode,
            "mode_entropy": mode_entropy,
        }

    @staticmethod
    def mode_names() -> list[str]:
        """Return list of 12 mode names."""
        return MODES.copy()
