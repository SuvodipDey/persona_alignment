import json

from fuzzywuzzy import fuzz

PERSONAS_FILE = "./personas_40_selected.json"
PARTIAL_RATIO_THRESHOLD = 95


def load_personas(path=PERSONAS_FILE):
    """Load personas from JSON and attach a 1-indexed persona_id to each entry."""
    with open(path, encoding="utf-8") as f:
        personas = json.load(f)
    for idx, persona in enumerate(personas, start=1):
        persona["persona_id"] = idx
    return personas


def match_persona(attributes, personas, threshold=PARTIAL_RATIO_THRESHOLD, consider_keys=None):
    """Return (persona_id, description) for the first persona whose opa_filters
    fuzzy-match attributes using partial_ratio >= threshold.

    consider_keys: if provided, only opa_filter keys in this set are checked.
    persona_id is 1-indexed. Returns (None, None) if no match found.
    """
    for persona in personas:
        filters = {
            k: v for k, v in persona["opa_filters"].items()
            if consider_keys is None or k in consider_keys
        }
        if filters and all(
            fuzz.partial_ratio(attributes.get(k, ""), v) >= threshold
            for k, v in filters.items()
        ):
            return persona["persona_id"], persona["description"]
    return None, None


def find_best_persona(attributes, personas, threshold=PARTIAL_RATIO_THRESHOLD,
                      consider_keys=None, min_passing=None):
    """Return (persona_id, description, n_passing, n_total, unmatched) for the
    persona with the most passing opa_filters, provided that count >= min_passing.
    Ties broken by order (first wins). If min_passing is None, all filters must pass.

    unmatched: {attr: {"expected": ..., "got": ...}} for each failing filter.
    Returns (None, None, 0, 0, {}) if no persona qualifies.
    """
    best_pid, best_desc, best_count, best_n_total, best_unmatched = None, None, -1, 0, {}
    for persona in personas:
        filters = {
            k: v for k, v in persona["opa_filters"].items()
            if consider_keys is None or k in consider_keys
        }
        if not filters:
            continue
        n_total = len(filters)
        n_passing = 0
        unmatched = {}
        for k, v in filters.items():
            if fuzz.partial_ratio(attributes.get(k, ""), v) >= threshold:
                n_passing += 1
            else:
                unmatched[k] = {"expected": v, "got": attributes.get(k, "")}
        required = min_passing if min_passing is not None else n_total
        if n_passing >= required and n_passing > best_count:
            best_count = n_passing
            best_pid = persona["persona_id"]
            best_desc = persona["description"]
            best_n_total = n_total
            best_unmatched = unmatched
    return best_pid, best_desc, max(best_count, 0), best_n_total, best_unmatched


def find_matching_personas(attributes, personas, threshold=PARTIAL_RATIO_THRESHOLD,
                           consider_keys=None, min_passing=None):
    """Return a list of (persona_id, description) for every persona where the
    number of passing opa_filters >= min_passing.

    If min_passing is None, all filters must pass (equivalent to match_persona
    but returning all matches instead of just the first).
    consider_keys: if provided, only opa_filter keys in this set are checked.
    """
    matches = []
    for persona in personas:
        filters = {
            k: v for k, v in persona["opa_filters"].items()
            if consider_keys is None or k in consider_keys
        }
        if not filters:
            continue
        n_passing = sum(
            1 for k, v in filters.items()
            if fuzz.partial_ratio(attributes.get(k, ""), v) >= threshold
        )
        required = min_passing if min_passing is not None else len(filters)
        if n_passing >= required:
            matches.append((persona["persona_id"], persona["description"]))
    return matches
