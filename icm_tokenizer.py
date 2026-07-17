"""
Intensional Conceptual Model (ICM) tokenizer.
Parses ask text into flat token sequences with explicit START/END boundaries.
"""

import re
from dataclasses import dataclass
from typing import Literal
import spacy

TokenType = Literal["Entity", "Action", "Property", "Goal", "Context", "Boundary"]


@dataclass
class Token:
    """A single token in the ICM parse."""
    text: str
    type: TokenType
    group_id: int  # which compound group this belongs to
    group_type: str  # "Entity", "Action", "Property", etc.
    is_boundary: bool = False  # START or END marker


class ICMTokenizer:
    def __init__(self):
        """Initialize spaCy model for POS tagging."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )

    def tokenize(self, ask_text: str) -> list[Token]:
        """
        Parse ask text into flat token sequence with boundaries.

        Example input:
            "I want to build an AI that understands coherence for research teams"

        Returns token list with START/END markers:
            [START Entity:I], I, [END Entity:I], [START Action:want], want, ...
        """
        # Clean and normalize
        ask_text = ask_text.strip()

        # Process with spaCy
        doc = self.nlp(ask_text)

        # Identify token groups (Subject, Action, Object, etc.)
        groups = self._identify_groups(doc)

        # Flatten into token sequence with boundaries
        tokens = self._flatten_groups(groups, doc)

        return tokens

    def _identify_groups(self, doc) -> list[dict]:
        """
        Identify conceptual groups from spaCy doc.
        Returns: [{"type": "Entity", "tokens": [token_idx, ...], "head": idx}, ...]
        """
        groups = []
        group_id = 0

        # Find ROOT verb (main action)
        root_verb = None
        for token in doc:
            if token.dep_ == "ROOT":
                root_verb = token
                break

        if not root_verb:
            root_verb = doc[0]

        # Subject (typically nsubj of ROOT)
        subject_tokens = []
        for token in doc:
            if token.head == root_verb and token.dep_ in ("nsubj", "nsubjpass"):
                subject_tokens.append(token.i)

        if subject_tokens:
            groups.append(
                {"type": "Entity", "tokens": subject_tokens, "head": subject_tokens[0], "id": group_id}
            )
            group_id += 1

        # Action (ROOT verb + auxiliaries)
        action_tokens = [root_verb.i]
        for token in root_verb.children:
            if token.dep_ in ("aux", "auxpass"):
                action_tokens.append(token.i)
        action_tokens.sort()
        groups.append(
            {"type": "Action", "tokens": action_tokens, "head": root_verb.i, "id": group_id}
        )
        group_id += 1

        # Direct object (typically dobj of ROOT)
        dobj_tokens = []
        for token in doc:
            if token.head == root_verb and token.dep_ in ("dobj", "attr"):
                dobj_tokens.append(token.i)
                # Include adjectival modifiers (amod)
                for child in token.subtree:
                    if child.dep_ == "amod":
                        dobj_tokens.append(child.i)

        if dobj_tokens:
            dobj_tokens.sort()
            groups.append(
                {"type": "Entity", "tokens": dobj_tokens, "head": dobj_tokens[0], "id": group_id}
            )
            group_id += 1

        # Relative clauses and other actions (acl, advcl)
        for token in doc:
            if token.pos_ == "VERB" and token != root_verb:
                clause_tokens = [token.i]
                # Include direct object of this verb
                for child in token.children:
                    if child.dep_ in ("dobj", "attr"):
                        clause_tokens.append(child.i)
                        # Include modifiers
                        for grandchild in child.subtree:
                            if grandchild.dep_ == "amod":
                                clause_tokens.append(grandchild.i)
                if clause_tokens:
                    clause_tokens.sort()
                    groups.append(
                        {
                            "type": "Action",
                            "tokens": clause_tokens,
                            "head": token.i,
                            "id": group_id,
                        }
                    )
                    group_id += 1

        # Prepositional phrases (for, in, etc.) → Context
        for token in doc:
            if token.pos_ == "ADP":  # Preposition
                pp_tokens = [token.i]
                # Include everything in the PP
                for child in token.children:
                    pp_tokens.append(child.i)
                    for grandchild in child.subtree:
                        if grandchild != child:
                            pp_tokens.append(grandchild.i)
                if pp_tokens:
                    pp_tokens.sort()
                    groups.append(
                        {
                            "type": "Context",
                            "tokens": pp_tokens,
                            "head": token.i,
                            "id": group_id,
                        }
                    )
                    group_id += 1

        # Merge overlapping groups (prefer larger groups)
        groups = self._merge_overlapping_groups(groups)

        # Identify ungrouped tokens as properties/modifiers
        grouped_token_indices = set()
        for group in groups:
            grouped_token_indices.update(group["tokens"])

        for i, token in enumerate(doc):
            if i not in grouped_token_indices:
                if token.pos_ in ("ADJ", "ADV"):
                    # Find the nearest noun/verb to attach to
                    groups.append(
                        {
                            "type": "Property",
                            "tokens": [i],
                            "head": i,
                            "id": group_id,
                        }
                    )
                    group_id += 1

        # Sort groups by first token position
        groups.sort(key=lambda g: min(g["tokens"]))

        return groups

    def _merge_overlapping_groups(self, groups: list[dict]) -> list[dict]:
        """Remove groups that overlap with larger groups."""
        merged = []
        for i, g1 in enumerate(groups):
            overlaps = False
            for j, g2 in enumerate(groups):
                if i != j:
                    overlap = set(g1["tokens"]) & set(g2["tokens"])
                    if overlap and len(g2["tokens"]) > len(g1["tokens"]):
                        overlaps = True
                        break
            if not overlaps:
                merged.append(g1)
        return merged

    def _flatten_groups(self, groups: list[dict], doc) -> list[Token]:
        """Convert groups into flat token list with START/END boundaries."""
        tokens = []

        for group in groups:
            group_id = group["id"]
            group_type = group["type"]
            token_indices = sorted(group["tokens"])

            # START boundary
            boundary_text = f"[START {group_type}:{group_id}]"
            tokens.append(
                Token(
                    text=boundary_text,
                    type="Boundary",
                    group_id=group_id,
                    group_type=group_type,
                    is_boundary=True,
                )
            )

            # Tokens in group
            for idx in token_indices:
                token = doc[idx]
                tokens.append(
                    Token(
                        text=token.text,
                        type=group_type,
                        group_id=group_id,
                        group_type=group_type,
                    )
                )

            # END boundary
            boundary_text = f"[END {group_type}:{group_id}]"
            tokens.append(
                Token(
                    text=boundary_text,
                    type="Boundary",
                    group_id=group_id,
                    group_type=group_type,
                    is_boundary=True,
                )
            )

        return tokens
