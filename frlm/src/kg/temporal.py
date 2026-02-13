"""
Temporal Resolver — Temporal filtering and consistency validation for FRLM facts.

Provides:
    - resolve_current: filter to currently valid facts (valid_to is None)
    - resolve_at: filter to facts valid at a specific timestamp
    - resolve_history: return all versions of facts ordered by valid_from
    - validate_temporal_consistency: check for overlapping windows and chain gaps
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional

from src.kg.schema import Fact

logger = logging.getLogger(__name__)


class TemporalResolver:
    """Resolves temporal queries over lists of facts.

    This class implements the three temporal modes used by the FRLM
    retrieval head:

    - **CURRENT**: Only facts with ``valid_to is None`` (not superseded).
    - **AT_TIMESTAMP**: Facts whose validity window contains the timestamp.
    - **HISTORY**: All versions of each fact, ordered chronologically.

    It also provides validation to detect integrity issues in the
    temporal model (overlapping windows, gaps in version chains).
    """

    # -----------------------------------------------------------------
    # Core resolution methods
    # -----------------------------------------------------------------

    @staticmethod
    def resolve_current(facts: List[Fact]) -> List[Fact]:
        """Return only currently valid facts.

        A fact is current iff ``valid_to is None`` (it has not been
        superseded).

        Parameters
        ----------
        facts : list of Fact
            Input facts (any temporal state).

        Returns
        -------
        list of Fact
            Facts where ``temporal.valid_to is None``.
        """
        current = [f for f in facts if f.temporal.is_current]
        logger.debug(
            "resolve_current: %d / %d facts are current",
            len(current), len(facts),
        )
        return current

    @staticmethod
    def resolve_at(facts: List[Fact], timestamp: date) -> List[Fact]:
        """Return facts valid at a specific timestamp.

        A fact is valid at *timestamp* iff:
        ``valid_from <= timestamp`` and
        (``valid_to is None`` or ``timestamp < valid_to``).

        Parameters
        ----------
        facts : list of Fact
            Input facts.
        timestamp : date
            The point-in-time to query.

        Returns
        -------
        list of Fact
            Facts valid at the given timestamp.
        """
        valid = [f for f in facts if f.temporal.contains(timestamp)]
        logger.debug(
            "resolve_at(%s): %d / %d facts valid",
            timestamp.isoformat(), len(valid), len(facts),
        )
        return valid

    @staticmethod
    def resolve_history(facts: List[Fact]) -> List[Fact]:
        """Return all versions of facts ordered by valid_from (oldest first).

        No filtering is applied — all temporal versions are returned.

        Parameters
        ----------
        facts : list of Fact
            Input facts (any temporal state).

        Returns
        -------
        list of Fact
            All facts sorted by ``temporal.valid_from`` ascending.
        """
        ordered = sorted(facts, key=lambda f: f.temporal.valid_from)
        logger.debug("resolve_history: %d facts ordered chronologically", len(ordered))
        return ordered

    # -----------------------------------------------------------------
    # Convenience dispatcher
    # -----------------------------------------------------------------

    def resolve(
        self,
        facts: List[Fact],
        mode: str,
        timestamp: Optional[date] = None,
    ) -> List[Fact]:
        """Dispatch to the appropriate resolution method.

        Parameters
        ----------
        facts : list of Fact
            Input facts.
        mode : str
            One of ``"CURRENT"``, ``"AT_TIMESTAMP"``, ``"HISTORY"``.
        timestamp : date, optional
            Required when mode is ``"AT_TIMESTAMP"``.

        Returns
        -------
        list of Fact

        Raises
        ------
        ValueError
            If the mode is unknown or timestamp is missing for AT_TIMESTAMP.
        """
        mode_upper = mode.upper()

        if mode_upper == "CURRENT":
            return self.resolve_current(facts)
        if mode_upper == "AT_TIMESTAMP":
            if timestamp is None:
                raise ValueError("timestamp is required for AT_TIMESTAMP mode")
            return self.resolve_at(facts, timestamp)
        if mode_upper == "HISTORY":
            return self.resolve_history(facts)

        raise ValueError(
            f"Unknown temporal mode '{mode}'. "
            f"Must be one of: CURRENT, AT_TIMESTAMP, HISTORY"
        )

    # -----------------------------------------------------------------
    # Temporal consistency validation
    # -----------------------------------------------------------------

    @staticmethod
    def validate_temporal_consistency(facts: List[Fact]) -> List[str]:
        """Check for temporal integrity issues across a set of facts.

        Groups facts by ``family_key`` (same logical triple, different
        temporal versions) and checks for:

        1. **Overlapping validity windows** — two versions of the same
           fact whose time ranges overlap.
        2. **Gaps in version chains** — a superseded fact's ``valid_to``
           does not match the next version's ``valid_from``.
        3. **Multiple current versions** — more than one version of the
           same logical fact has ``valid_to is None``.
        4. **Inverted windows** — ``valid_from >= valid_to`` (caught by
           Pydantic, but checked here for data loaded from external sources).

        Parameters
        ----------
        facts : list of Fact
            All facts to validate (may span multiple family keys).

        Returns
        -------
        list of str
            Human-readable error messages. Empty list means all valid.
        """
        errors: List[str] = []

        # Group by family_key
        families: Dict[str, List[Fact]] = defaultdict(list)
        for fact in facts:
            families[fact.family_key].append(fact)

        for family_key, family_facts in families.items():
            family_errors = TemporalResolver._validate_family(family_key, family_facts)
            errors.extend(family_errors)

        if errors:
            logger.warning(
                "Temporal consistency check found %d issue(s)", len(errors)
            )
        else:
            logger.debug(
                "Temporal consistency check passed for %d facts across %d families",
                len(facts), len(families),
            )

        return errors

    @staticmethod
    def _validate_family(family_key: str, facts: List[Fact]) -> List[str]:
        """Validate a single family of temporal fact versions."""
        errors: List[str] = []
        fk_short = family_key[:16]

        # Sort by valid_from for chronological analysis
        sorted_facts = sorted(facts, key=lambda f: f.temporal.valid_from)

        # Check 1: Multiple current versions
        current_facts = [f for f in sorted_facts if f.temporal.is_current]
        if len(current_facts) > 1:
            ids = [f.fact_id[:16] for f in current_facts]
            errors.append(
                f"Family {fk_short}...: {len(current_facts)} current versions "
                f"(expected at most 1). Fact IDs: {ids}"
            )

        # Check 2: Inverted windows (valid_from >= valid_to)
        for fact in sorted_facts:
            if (
                fact.temporal.valid_to is not None
                and fact.temporal.valid_from >= fact.temporal.valid_to
            ):
                errors.append(
                    f"Family {fk_short}...: Fact {fact.fact_id[:16]}... has "
                    f"inverted window: valid_from={fact.temporal.valid_from} "
                    f">= valid_to={fact.temporal.valid_to}"
                )

        # Check 3: Overlapping windows
        for i in range(len(sorted_facts)):
            for j in range(i + 1, len(sorted_facts)):
                fi = sorted_facts[i]
                fj = sorted_facts[j]

                if _windows_overlap(fi, fj):
                    errors.append(
                        f"Family {fk_short}...: Overlapping windows between "
                        f"{fi.fact_id[:16]}... "
                        f"[{fi.temporal.valid_from}, {fi.temporal.valid_to}] and "
                        f"{fj.fact_id[:16]}... "
                        f"[{fj.temporal.valid_from}, {fj.temporal.valid_to}]"
                    )

        # Check 4: Gaps in version chain
        # Only applicable to superseded facts followed by another version
        closed_facts = [f for f in sorted_facts if not f.temporal.is_current]
        for closed in closed_facts:
            # Find the next version (valid_from == this fact's valid_to)
            expected_next_from = closed.temporal.valid_to
            next_versions = [
                f for f in sorted_facts
                if f.temporal.valid_from == expected_next_from
                and f.fact_id != closed.fact_id
            ]
            if not next_versions and expected_next_from is not None:
                # Check if any version starts after this closes
                later = [
                    f for f in sorted_facts
                    if f.temporal.valid_from > closed.temporal.valid_from
                    and f.fact_id != closed.fact_id
                ]
                if later:
                    nearest = min(later, key=lambda f: f.temporal.valid_from)
                    if nearest.temporal.valid_from != expected_next_from:
                        errors.append(
                            f"Family {fk_short}...: Gap in version chain — "
                            f"fact {closed.fact_id[:16]}... ends at "
                            f"{closed.temporal.valid_to} but next version "
                            f"{nearest.fact_id[:16]}... starts at "
                            f"{nearest.temporal.valid_from}"
                        )

        return errors


# ===========================================================================
# Helpers
# ===========================================================================


def _windows_overlap(f1: Fact, f2: Fact) -> bool:
    """Return True if two facts have overlapping validity windows.

    Two half-open intervals [a, b) and [c, d) overlap iff a < d and c < b.
    When b or d is None (open-ended), treat it as +infinity.
    """
    a = f1.temporal.valid_from
    b = f1.temporal.valid_to  # None = +inf
    c = f2.temporal.valid_from
    d = f2.temporal.valid_to  # None = +inf

    # a < d (or d is None => always true)
    a_lt_d = d is None or a < d
    # c < b (or b is None => always true)
    c_lt_b = b is None or c < b

    return a_lt_d and c_lt_b
