"""
Build weighted adjacency matrix from ICM tokens.
"""

import numpy as np
from icm_tokenizer import Token, TokenType


class MatrixBuilder:
    """Convert token sequence to weighted adjacency matrix."""

    # Edge weights by relationship type
    WEIGHTS = {
        "intra_group": 0.9,      # token ↔ token within same group
        "group_boundary": 0.8,    # START/END ↔ tokens in group
        "inter_group": 0.6,       # END[group1] ↔ START[group2]
        "modifier": 0.7,          # property ↔ entity/action
        "context": 0.4,           # context attachment
    }

    def build(self, tokens: list[Token]) -> tuple[np.ndarray, list[str]]:
        """
        Build weighted adjacency matrix from tokens.

        Returns:
            (adjacency_matrix: np.ndarray, node_names: list[str])
        """
        n = len(tokens)
        matrix = np.zeros((n, n), dtype=np.float32)
        node_names = [f"{t.text}({t.group_id})" for t in tokens]

        # Add edges
        for i in range(n):
            for j in range(i + 1, n):
                weight = self._edge_weight(tokens, i, j)
                if weight > 0:
                    # Symmetric (undirected)
                    matrix[i, j] = weight
                    matrix[j, i] = weight

        return matrix, node_names

    def _edge_weight(self, tokens: list[Token], i: int, j: int) -> float:
        """Determine edge weight between tokens i and j."""
        ti = tokens[i]
        tj = tokens[j]

        # Both are boundaries
        if ti.is_boundary and tj.is_boundary:
            return 0

        # One is a boundary
        if ti.is_boundary or tj.is_boundary:
            return self._boundary_weight(tokens, i, j)

        # Both are regular tokens in same group (intra-group)
        if ti.group_id == tj.group_id and ti.group_type == tj.group_type:
            return self.WEIGHTS["intra_group"]

        # Adjacent tokens in sequence (loose connection)
        if abs(i - j) == 1 and not ti.is_boundary and not tj.is_boundary:
            # If different groups, this is inter-group
            if ti.group_id != tj.group_id:
                return self.WEIGHTS["inter_group"]

        # Property attached to entity
        if (ti.type == "Property" and tj.type == "Entity") or (
            ti.type == "Entity" and tj.type == "Property"
        ):
            if abs(i - j) <= 3:  # Within reasonable distance
                return self.WEIGHTS["modifier"]

        # Context attachment to action or entity
        if (ti.type == "Context" and tj.type in ("Action", "Entity")) or (
            ti.type in ("Action", "Entity") and tj.type == "Context"
        ):
            if abs(i - j) <= 5:
                return self.WEIGHTS["context"]

        return 0

    def _boundary_weight(self, tokens: list[Token], i: int, j: int) -> float:
        """Edge weight involving a boundary marker."""
        boundary_idx = i if tokens[i].is_boundary else j
        token_idx = j if tokens[i].is_boundary else i
        boundary = tokens[boundary_idx]
        token = tokens[token_idx]

        # START/END belongs to same group
        if boundary.group_id == token.group_id:
            return self.WEIGHTS["group_boundary"]

        # END[group1] → START[group2] (inter-group transition)
        if boundary.text.startswith("[END") and tokens[token_idx].text.startswith("[START"):
            if boundary.group_id < token.group_id:
                return self.WEIGHTS["inter_group"]

        return 0
