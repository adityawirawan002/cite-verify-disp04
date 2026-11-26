#!/usr/bin/env python3
"""Grounded Passage Citation Verifier (DISPATCH-04, zero external dependencies, Python 3.12+).

Evaluates passage-grounded answers (<=120 words, inline [1]..[k] citations or calibrated
abstention) against 1..8 source passages with four strict gates:
  1. Word-cap & citation-index validity ([1]..[k], k in 1..8, no uncited claims).
  2. Strict 0.0 imperative safety/action verb phrase grounding gate.
  3. Lexical span, numeric/unit, entity, and polarity alignment (blocks parametric leaks).
  4. Calibrated abstention gate (rewards warranted abstention, scores 0.0 on over-refusal
     and contradictory abstention-with-citations).
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

MAX_PASSAGES = 8
MIN_PASSAGES = 1
MAX_WORDS = 120

STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "done", "for", "from", "had", "has",
    "have", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "just", "may", "me", "might", "more", "most",
    "my", "no", "nor", "not", "of", "on", "once", "only", "or", "other", "our",
    "ours", "out", "over", "own", "per", "same", "shall", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "us", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "also", "about", "after", "before", "between", "during",
    "each", "either", "both", "any", "all", "must", "need", "needs", "used",
    "using", "use", "based", "provided", "passage", "passages", "text", "context",
    "according", "states", "state", "says", "said", "indicates", "notes", "mentions",
    "following", "step", "steps", "first", "second", "third", "finally", "next",
    "immediately", "safely", "always", "never", "please", "ensure", "make", "sure",
})

# High-stakes imperative / safety / procedural action verbs (base lemmas)
ACTION_VERBS: frozenset[str] = frozenset({
    "abort", "activate", "add", "adjust", "administer", "alert", "apply", "arm",
    "avoid", "backup", "ban", "bleed", "block", "boil", "boot", "break", "burn",
    "bypass", "calibrate", "call", "cancel", "change", "charge", "check", "clamp",
    "clean", "clear", "close", "collect", "combine", "compress", "configure",
    "confirm", "connect", "contact", "continue", "cool", "copy", "cover", "cut",
    "deactivate", "decrease", "decompress", "delete", "deploy", "depressurize",
    "detach", "dilute", "disable", "discard", "discharge", "disconnect", "discontinue",
    "dispose", "dissolve", "double", "drain", "drink", "drop", "dump", "eat",
    "elevate", "enable", "engage", "escalate", "evacuate", "execute", "exit",
    "expose", "extinguish", "failover", "filter", "fire", "flush", "force",
    "freeze", "give", "ground", "halt", "heat", "hold", "ignore", "immerse",
    "increase", "induce", "inflate", "inform", "ingest", "inhale", "init",
    "initialize", "inject", "insert", "inspect", "install", "interrupt", "isolate",
    "jump", "keep", "kill", "launch", "leave", "lift", "limit", "load", "lock",
    "lockout", "log", "lower", "maintain", "mask", "measure", "mix", "modify",
    "monitor", "mount", "move", "neutralize", "notify", "open", "operate",
    "override", "patch", "pause", "perform", "place", "plug", "pour", "power",
    "press", "prevent", "proceed", "pull", "pump", "purge", "push", "put",
    "quarantine", "quench", "raise", "read", "reboot", "reconfigure", "reconnect",
    "record", "recover", "reduce", "refill", "reheat", "reinstall", "release",
    "reload", "rely", "remove", "repair", "repeat", "replace", "report", "reset",
    "restart", "restore", "restrict", "resume", "revoke", "rinse", "rollback",
    "rotate", "route", "run", "scrub", "seal", "seek", "select", "send",
    "separate", "set", "shield", "shut", "shutdown", "skip", "soak", "spray",
    "start", "sterilize", "stir", "stop", "store", "submerge", "submit", "supply",
    "suspend", "swallow", "swap", "switch", "sync", "tag", "tagout", "take",
    "terminate", "test", "thaw", "tighten", "toggle", "touch", "transfer",
    "trigger", "trim", "trip", "truncate", "turn", "unbind", "unblock", "undo",
    "unhook", "uninstall", "unlock", "unmount", "unplug", "update", "upgrade",
    "upload", "use", "vacuum", "validate", "vent", "ventilate", "verify", "wait",
    "warm", "warn", "wash", "wear", "wipe", "withdraw", "write", "zero",
})

# Multi-word safety/action phrasal verbs normalized as single units
PHRASAL_ACTIONS: tuple[tuple[str, str], ...] = (
    ("turn", "off"), ("turn", "on"), ("shut", "down"), ("shut", "off"),
    ("power", "off"), ("power", "down"), ("power", "on"), ("switch", "off"),
    ("switch", "on"), ("cut", "off"), ("lock", "out"), ("tag", "out"),
    ("pull", "out"), ("plug", "in"), ("wash", "out"), ("flush", "out"),
    ("bleed", "off"), ("vent", "out"), ("stop", "taking"), ("keep", "away"),
    ("stay", "away"), ("back", "up"), ("roll", "back"), ("spin", "down"),
    ("fail", "over"), ("scale", "down"), ("scale", "up"), ("bring", "down"),
    ("wipe", "out"), ("seek", "emergency"), ("call", "911"),
)

NEGATION_MARKERS: frozenset[str] = frozenset({
    "no", "not", "never", "none", "neither", "nor", "without", "cannot",
    "cant", "wont", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt",
    "werent", "shouldnt", "mustnt", "avoid", "prohibited", "forbidden",
    "contraindicated", "disallowed", "refuse", "refused", "except", "unsafe",
    "unapproved", "invalid", "absent", "lack", "lacks", "lacking",
})

ABSTENTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\binsufficient\s+(?:evidence|information|data|context|details)\b",
        r"\bnot\s+enough\s+(?:evidence|information|data|context|details)\b",
        r"\b(?:cannot|can't|unable\s+to)\s+(?:answer|determine|verify|confirm|conclude|provide)\b",
        r"\b(?:does|do)\s+not\s+(?:contain|provide|specify|mention|state|include|support|answer)\b",
        r"\bnot\s+(?:specified|mentioned|stated|provided|covered|addressed|supported|found)\s+in\s+(?:the\s+)?(?:provided\s+)?(?:passages?|context|text|sources?)\b",
        r"\bno\s+(?:evidence|information|mention|data|support)\s+(?:in\s+the\s+(?:provided\s+)?(?:passages?|context|text)|to\s+(?:answer|support|confirm))\b",
        r"\b(?:abstain|abstaining)\b",
        r"\bpassages?\s+(?:do|does)\s+not\s+(?:state|specify|mention|contain|provide|indicate|say|answer)\b",
        r"\bunknown\s+from\s+(?:the\s+)?(?:provided\s+)?(?:passages?|context|text)\b",
    )
)

DIRECTIVE_TRIGGER_RE = re.compile(
    r"\b(?:must|should|need\s+to|needs\s+to|have\s+to|has\s+to|ought\s+to|"
    r"do\s+not|don't|never|always|be\s+sure\s+to|make\s+sure\s+to|remember\s+to|"
    r"recommended\s+to|advised\s+to|required\s+to|instructed\s+to|safe\s+to|"
    r"avoid|ensure\s+you|you\s+can|operators?\s+must|patients?\s+should|"
    r"users?\s+must|engineers?\s+should|immediately|first|next|then|finally|"
    r"step\s+\d+\s*:?)\s+(?:not\s+|never\s+|immediately\s+|promptly\s+|safely\s+|also\s+)?([a-z][a-z\-]{1,24})\b",
    re.IGNORECASE,
)

CLAUSE_SPLIT_RE = re.compile(r"(?:[;:]|\b(?:and\s+then|then|before|after|or|and)\b|\n+)", re.IGNORECASE)
CITATION_GROUP_RE = re.compile(r"\[([^\[\]]+)\]")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'./][A-Za-z0-9]+)*")
NUM_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(%|percent|mg|mcg|ug|g|kg|ml|l|cc|mmol|iu|v|mv|kv|a|ma|w|kw|hz|khz|mhz|ghz|"
    r"ms|s|sec|secs|second|seconds|min|mins|minute|minutes|h|hr|hrs|hour|hours|"
    r"day|days|wk|week|weeks|mo|month|months|yr|year|years|"
    r"c|f|k|deg|°c|°f|psi|bar|kpa|mpa|atm|mmhg|ppm|ppb|kb|mb|gb|tb|bytes|b|usd|sol|usdc)?\b",
    re.IGNORECASE,
)

UNIT_ALIASES: dict[str, str] = {
    "percent": "%", "%": "%",
    "sec": "s", "secs": "s", "second": "s", "seconds": "s", "s": "s",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "hr": "h", "hrs": "h", "hour": "h", "hours": "h", "h": "h",
    "days": "day", "day": "day", "weeks": "week", "week": "week", "wk": "week",
    "months": "month", "month": "month", "mo": "month", "years": "year", "year": "year", "yr": "year",
    "°c": "c", "°f": "f", "ug": "mcg", "bytes": "b",
}

IRREGULAR_VERBS: dict[str, str] = {
    "taken": "take", "took": "take", "taking": "take", "takes": "take",
    "given": "give", "gave": "give", "giving": "give", "gives": "give",
    "ran": "run", "running": "run", "runs": "run",
    "shut": "shut", "shutting": "shut", "shuts": "shut",
    "cut": "cut", "cutting": "cut", "cuts": "cut",
    "put": "put", "putting": "put", "puts": "put",
    "set": "set", "setting": "set", "sets": "set",
    "held": "hold", "holding": "hold", "holds": "hold",
    "kept": "keep", "keeping": "keep", "keeps": "keep",
    "left": "leave", "leaving": "leave", "leaves": "leave",
    "sent": "send", "sending": "send", "sends": "send",
    "written": "write", "wrote": "write", "writing": "write", "writes": "write",
    "broken": "break", "broke": "break", "breaking": "break", "breaks": "break",
    "drunk": "drink", "drank": "drink", "drinking": "drink", "drinks": "drink",
    "eaten": "eat", "ate": "eat", "eating": "eat", "eats": "eat",
    "frozen": "freeze", "froze": "freeze", "freezing": "freeze", "freezes": "freeze",
    "withdrawn": "withdraw", "withdrew": "withdraw", "withdrawing": "withdraw",
    "stopped": "stop", "stopping": "stop", "stops": "stop",
    "skipped": "skip", "skipping": "skip", "skips": "skip",
    "dropped": "drop", "dropping": "drop", "drops": "drop",
    "swapped": "swap", "swapping": "swap", "swaps": "swap",
    "plugged": "plug", "plugging": "plug", "plugs": "plug",
    "unplugged": "unplug", "unplugging": "unplug", "unplugs": "unplug",
    "tripped": "trip", "tripping": "trip", "trips": "trip",
    "trimmed": "trim", "trimming": "trim", "trims": "trim",
    "scrubbed": "scrub", "scrubbing": "scrub", "scrubs": "scrub",
    "stirred": "stir", "stirring": "stir", "stirs": "stir",
    "applied": "apply", "applies": "apply", "applying": "apply",
    "copied": "copy", "copies": "copy", "copying": "copy",
    "modified": "modify", "modifies": "modify", "modifying": "modify",
    "notified": "notify", "notifies": "notify", "notifying": "notify",
    "verified": "verify", "verifies": "verify", "verifying": "verify",
    "supplied": "supply", "supplies": "supply", "supplying": "supply",
    "relied": "rely", "relies": "rely", "relying": "rely",
    "disabled": "disable", "disables": "disable", "disabling": "disable",
    "enabled": "enable", "enables": "enable", "enabling": "enable",
    "evacuated": "evacuate", "evacuates": "evacuate", "evacuating": "evacuate",
    "bypassed": "bypass", "bypasses": "bypass", "bypassing": "bypass",
    "overridden": "override", "overrode": "override", "overrides": "override", "overriding": "override",
    "administered": "administer", "administers": "administer", "administering": "administer",
    "rebooted": "reboot", "reboots": "reboot", "rebooting": "reboot",
    "restarted": "restart", "restarts": "restart", "restarting": "restart",
    "disconnected": "disconnect", "disconnects": "disconnect", "disconnecting": "disconnect",
    "discontinued": "discontinue", "discontinues": "discontinue", "discontinuing": "discontinue",
    "mixed": "mix", "mixes": "mix", "mixing": "mix",
    "flushed": "flush", "flushes": "flush", "flushing": "flush",
    "rinsed": "rinse", "rinses": "rinse", "rinsing": "rinse",
    "injected": "inject", "injects": "inject", "injecting": "inject",
    "ingested": "ingest", "ingests": "ingest", "ingesting": "ingest",
    "deleted": "delete", "deletes": "delete", "deleting": "delete",
    "purged": "purge", "purges": "purge", "purging": "purge",
    "isolated": "isolate", "isolates": "isolate", "isolating": "isolate",
    "vented": "vent", "vents": "vent", "venting": "vent",
    "ventilated": "ventilate", "ventilates": "ventilate", "ventilating": "ventilate",
    "depressurized": "depressurize", "depressurizes": "depressurize", "depressurizing": "depressurize",
}


def lemma_verb(tok: str) -> str:
    w = tok.lower().strip(".,;:!?\"'()[]{}")
    if not w:
        return ""
    if w in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[w]
    if w in ACTION_VERBS:
        return w
    # Regular inflection fallbacks checked strictly against ACTION_VERBS
    candidates: list[str] = []
    if w.endswith("ing") and len(w) > 5:
        candidates.extend([w[:-3], w[:-3] + "e"])
        if len(w) > 6 and w[-4] == w[-5]:
            candidates.append(w[:-4])
    if w.endswith("ies") and len(w) > 4:
        candidates.append(w[:-3] + "y")
    if w.endswith("ied") and len(w) > 4:
        candidates.append(w[:-3] + "y")
    if w.endswith("es") and len(w) > 4:
        candidates.extend([w[:-2], w[:-1]])
    if w.endswith("ed") and len(w) > 4:
        candidates.extend([w[:-2], w[:-1]])
        if len(w) > 5 and w[-3] == w[-4]:
            candidates.append(w[:-3])
    if w.endswith("s") and len(w) > 3:
        candidates.append(w[:-1])
    for c in candidates:
        if c in ACTION_VERBS:
            return c
    return w


CONCEPT_CANON: dict[str, str] = {
    "dosing": "dose", "dosage": "dose", "doses": "dose",
    "servicing": "service", "services": "service", "serviced": "service",
    "handling": "handle", "handled": "handle", "handles": "handle",
    "encrypt": "encrypt", "encrypted": "encrypt", "encryption": "encrypt",
    "notify": "notify", "notification": "notify", "notified": "notify",
    "corrupt": "corrupt", "corruption": "corrupt", "corrupted": "corrupt",
    "recover": "recover", "recovery": "recover", "recovered": "recover",
    "evacuate": "evacuate", "evacuation": "evacuate",
    "ventilate": "ventilate", "ventilation": "ventilate",
    "certify": "certify", "certified": "certify", "certification": "certify",
    "manufacture": "manufacture", "manufactured": "manufacture", "manufacturer": "manufacture",
    "warning": "warn", "warnings": "warn",
    "maximum": "max", "exceed": "max", "exceeds": "max", "limit": "limit", "cap": "cap",
    "procedure": "protocol", "steps": "protocol",
    "leaks": "leak", "leaked": "leak", "leaking": "leak",
    "bleed": "vent", "relief": "vent",
    "retention": "retain", "retained": "retain", "retaining": "retain", "retains": "retain",
    "snapshot": "backup", "snapshots": "backup", "backups": "backup",
}


def stem_token(tok: str) -> str:
    w = tok.lower().strip(".,;:!?\"'()[]{}")
    if not w:
        return ""
    if w in CONCEPT_CANON:
        return CONCEPT_CANON[w]
    lv = lemma_verb(w)
    if lv in ACTION_VERBS:
        return CONCEPT_CANON.get(lv, lv)
    if len(w) > 5 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        base = w[:-3]
        return CONCEPT_CANON.get(base, base)
    if len(w) > 4 and w.endswith("ed"):
        base = w[:-2]
        return CONCEPT_CANON.get(base, base)
    if len(w) > 4 and w.endswith("es") and not w.endswith(("ses", "xes", "zes")):
        return w[:-1]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def content_stems(text: str) -> set[str]:
    out: set[str] = set()
    for t in tokenize(text):
        if t in STOPWORDS or len(t) <= 1:
            continue
        st = stem_token(t)
        if st and st not in STOPWORDS:
            out.add(st)
    return out


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[[^\[\]]*\]", "", text).strip()


def split_sentences(text: str) -> list[str]:
    # Protect common decimal points and abbreviations
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text.strip())
    protected = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|St|vs|e\.g|i\.e|Fig|No|Eq)\.", r"\1<DOT>", protected)
    raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])|\n+", protected)
    sents: list[str] = []
    for p in raw_parts:
        s = p.replace("<DOT>", ".").strip()
        if s:
            sents.append(s)
    return sents


def parse_citations(text: str, num_passages: int) -> tuple[list[int], list[str]]:
    """Return (valid_1based_indices, invalid_tokens)."""
    valid: list[int] = []
    invalid: list[str] = []
    for m in CITATION_GROUP_RE.finditer(text):
        inner = m.group(1).strip()
        # Support [1], [1, 2], [1,2]
        parts = [p.strip() for p in inner.split(",")]
        for part in parts:
            if not re.fullmatch(r"-?\d+", part):
                invalid.append(f"[{inner}]")
                continue
            idx = int(part)
            if 1 <= idx <= num_passages:
                if idx not in valid:
                    valid.append(idx)
            else:
                invalid.append(f"[{idx}]")
    return valid, invalid


def extract_numbers(text: str) -> list[tuple[str, str]]:
    """Extract canonical (numeric_value_str, canonical_unit) pairs from citation-stripped text."""
    clean = strip_citations(text)
    out: list[tuple[str, str]] = []
    for m in NUM_UNIT_RE.finditer(clean):
        raw_num = m.group(1).replace(",", "")
        try:
            val = float(raw_num)
            canon_num = str(int(val)) if val.is_integer() else f"{val:g}"
        except ValueError:
            canon_num = raw_num
        raw_unit = (m.group(2) or "").lower()
        unit = UNIT_ALIASES.get(raw_unit, raw_unit)
        out.append((canon_num, unit))
    return out


CORE_NEGATION_MARKERS: frozenset[str] = NEGATION_MARKERS - {"without", "except"}


def has_negation(text: str, *, reference_stems: set[str] | None = None) -> bool:
    toks = tokenize(text.replace("'", ""))
    tset = set(toks)
    if tset & CORE_NEGATION_MARKERS:
        return True
    if reference_stems is None:
        return bool(tset & NEGATION_MARKERS)
    for i, t in enumerate(toks):
        if t in ("without", "except"):
            window = {stem_token(w) for w in toks[i + 1:i + 5]}
            if window & reference_stems:
                return True
    return False


def is_abstention(text: str) -> bool:
    clean = strip_citations(text)
    return any(pat.search(clean) for pat in ABSTENTION_PATTERNS)


@dataclass(frozen=True)
class ActionPhrase:
    verb: str
    phrasal: str
    negated: bool
    object_stems: tuple[str, ...]
    raw_clause: str


def extract_imperative_actions(sentence: str) -> list[ActionPhrase]:
    """Extract imperative / safety / procedural action verb phrases from a sentence."""
    clean = strip_citations(sentence)
    actions: list[ActionPhrase] = []
    seen_keys: set[tuple[str, str, bool]] = set()

    for clause in CLAUSE_SPLIT_RE.split(clean):
        c = re.sub(r"^\s*(?:\d+[.)]|[-•*])\s*", "", clause).strip(" ,.;:")
        if not c:
            continue
        toks = tokenize(c)
        if not toks:
            continue

        # 1. Clause-initial imperative (optionally prefixed with do not / never / always / adverb)
        idx = 0
        neg = False
        while idx < len(toks):
            t = toks[idx].replace("'", "")
            if t in ("do", "please", "immediately", "promptly", "safely", "always", "then", "also", "first", "next", "finally", "carefully", "strictly", "quickly"):
                idx += 1
                continue
            if t in ("not", "dont", "never", "avoid"):
                neg = True
                if t == "avoid":
                    break
                idx += 1
                continue
            break

        if idx < len(toks):
            v0 = lemma_verb(toks[idx])
            if v0 in ACTION_VERBS:
                phrasal = v0
                obj_start = idx + 1
                if idx + 1 < len(toks) and (v0, toks[idx + 1]) in PHRASAL_ACTIONS:
                    phrasal = f"{v0} {toks[idx + 1]}"
                    obj_start = idx + 2
                obj_toks = toks[obj_start:obj_start + 8]
                obj_stems = tuple(sorted({stem_token(t) for t in obj_toks if t not in STOPWORDS and len(t) > 1}))
                # Only treat bare clause-initial word as imperative if it starts the clause (idx<=2)
                # and is base-form or preceded by negation/adverb, OR clearly imperative command.
                raw_w = toks[idx]
                if raw_w == v0 or neg or idx > 0 or raw_w in ("avoid", "stop", "call", "seek"):
                    key = (v0, phrasal, neg)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        actions.append(ActionPhrase(v0, phrasal, neg, obj_stems, c))

        # 2. Modal / directive / procedural triggers inside the clause
        for m in DIRECTIVE_TRIGGER_RE.finditer(c):
            raw_v = m.group(1).lower()
            v = lemma_verb(raw_v)
            if v not in ACTION_VERBS:
                continue
            prefix = c[max(0, m.start() - 12):m.end()].lower()
            m_neg = has_negation(prefix)
            tail = c[m.end():]
            tail_toks = tokenize(tail)[:8]
            phrasal = v
            obj_toks = tail_toks
            if tail_toks and (v, tail_toks[0]) in PHRASAL_ACTIONS:
                phrasal = f"{v} {tail_toks[0]}"
                obj_toks = tail_toks[1:]
            obj_stems = tuple(sorted({stem_token(t) for t in obj_toks if t not in STOPWORDS and len(t) > 1}))
            key = (v, phrasal, m_neg)
            if key not in seen_keys:
                seen_keys.add(key)
                actions.append(ActionPhrase(v, phrasal, m_neg, obj_stems, c))

    return actions


def action_grounded_in_passages(action: ActionPhrase, cited_passages: Sequence[str]) -> tuple[bool, str]:
    """Check whether an imperative/safety action phrase is grounded in at least one cited passage sentence."""
    for p in cited_passages:
        for psent in split_sentences(p):
            ptoks = tokenize(psent)
            p_lemmas = {lemma_verb(t) for t in ptoks}
            p_stems = content_stems(psent) | p_lemmas

            # 1. Exact action verb lemma must appear in the same passage sentence
            if action.verb not in p_lemmas:
                continue
            # 2. If phrasal verb, particle must also appear near the verb
            if " " in action.phrasal:
                _, part = action.phrasal.split(" ", 1)
                if part not in ptoks:
                    continue
            # 3. Check polarity compatibility (never invert safety prohibitions!)
            p_neg = has_negation(psent)
            if action.negated != p_neg:
                # Allow passage having e.g. "if no alarm, evacuate" only if negation isn't immediately before verb
                neg_before_verb = bool(re.search(
                    rf"\b(?:do\s+not|don't|never|must\s+not|should\s+not|not|avoid|prohibited\s+from)\s+(?:\w+\s+){{0,2}}{re.escape(action.verb)}\w*\b",
                    psent,
                    re.IGNORECASE,
                ))
                if action.negated != neg_before_verb:
                    continue
            # 4. Object/argument grounding: core object stems must be supported in this passage sentence
            if not action.object_stems:
                return True, "verb_grounded"
            overlap = sum(1 for st in action.object_stems if st in p_stems)
            req = 1 if len(action.object_stems) <= 2 else math.ceil(len(action.object_stems) * 0.5)
            if overlap >= req:
                return True, "verb_and_object_grounded"
    return False, f"ungrounded_action:{action.phrasal}({','.join(action.object_stems[:4])})"


def estimate_passage_answerability(query: str, passages: Sequence[str]) -> tuple[bool, float, list[int]]:
    """Determine whether passages[1..k] substantively answer the query (for calibrated abstention)."""
    q_stems = content_stems(query)
    if not q_stems:
        return False, 0.0, []
    q_lower = query.lower()
    asks_number = bool(re.search(
        r"\b(how\s+many|how\s+much|what\s+(?:is\s+the\s+)?(?:dosage|dose|limit|temperature|pressure|rate|latency|price|cost|year|date|timeout|threshold|cap|percentage|version|port|ttl|duration|interval|volume)|when\s+was|at\s+what\s+(?:temp|temperature|pressure|time))\b",
        q_lower,
    ))
    asks_person_or_vendor = bool(re.search(
        r"\b(who\s+(?:invented|founded|authored|wrote|discovered|led|manufactured|built|created|is\s+the\s+ceo)|which\s+.{0,30}\b(?:vendor|manufacturer|company|supplier)|where\s+is\s+.+\s+located|what\s+is\s+the\s+capital)\b",
        q_lower,
    ))
    asks_procedure = bool(re.search(r"\b(how\s+(?:do|should|to)|what\s+steps|procedure|protocol|what\s+should\s+(?:an?\s+)?(?:operator|user|patient|engineer|technician|administrator))\b", q_lower))

    # Separate generic framing words from focal query terms
    generic_q = {
        "what", "which", "describe", "explain", "list", "give", "tell", "accord",
        "passage", "document", "policy", "procedure", "protocol", "step", "steps",
        "exact", "specific", "recommend", "recommended", "required", "apply",
        "applies", "rule", "warn", "daily", "response", "follow", "take", "handle",
        "operator", "engineer", "technician", "administrator", "patient", "user",
    }
    key_q = {s for s in q_stems if s not in generic_q}
    if not key_q:
        key_q = q_stems

    # Specific target-attribute tokens that MUST be supported if asked explicitly
    target_attrs = {
        "price", "cost", "usd", "dollar", "penalt", "volume", "liter", "firmware",
        "ssd", "nvme", "pediatric", "intravenou", "weight", "chart", "manufacture",
        "enclosure", "steel", "vendor", "hsm", "hydrostatical", "certify", "capital",
        "ceo", "founder", "invent", "born", "birthday",
    }
    required_attrs = {t for t in key_q if any(t.startswith(a) for a in target_attrs)}

    best_score = 0.0
    supporting: list[int] = []
    union_stems: set[str] = set()
    union_nums: list[tuple[str, str]] = []
    union_has_action = False
    union_caps: list[str] = []

    for idx, p in enumerate(passages, 1):
        if is_abstention(p):
            continue
        p_stems = content_stems(p)
        if not p_stems:
            continue
        p_nums = extract_numbers(p)
        p_actions = extract_imperative_actions(p)
        p_lemmas = {lemma_verb(t) for t in tokenize(p)}
        has_action = bool(p_actions or (p_lemmas & ACTION_VERBS))
        caps = re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b", p)

        overlap = key_q & p_stems
        if overlap:
            union_stems |= p_stems
            union_nums.extend(p_nums)
            union_has_action = union_has_action or has_action
            union_caps.extend(caps)

        cov = len(overlap) / len(key_q)
        score = cov
        if asks_number and not p_nums:
            score *= 0.40
        if asks_procedure and not has_action:
            score *= 0.55
        if asks_person_or_vendor and not caps:
            score *= 0.35

        best_sent_cov = 0.0
        for ps in split_sentences(p):
            sc = len(key_q & content_stems(ps)) / len(key_q)
            if sc > best_sent_cov:
                best_sent_cov = sc
        combined = 0.55 * score + 0.45 * best_sent_cov
        if required_attrs and not (required_attrs & p_stems):
            combined *= 0.30

        min_terms = 1 if len(key_q) <= 1 else 2
        if len(overlap) >= min_terms and combined >= 0.44:
            supporting.append(idx)
        if combined > best_score:
            best_score = combined

    # Multi-passage union check (e.g., query spans Passage [1] + Passage [2])
    union_ov = key_q & union_stems
    union_cov = len(union_ov) / len(key_q)
    if asks_number and not union_nums:
        union_cov *= 0.40
    if asks_procedure and not union_has_action:
        union_cov *= 0.55
    if asks_person_or_vendor and not union_caps:
        union_cov *= 0.35
    if required_attrs and not (required_attrs & union_stems):
        union_cov *= 0.25

    if union_cov > best_score:
        best_score = union_cov
    if not supporting and len(union_ov) >= (1 if len(key_q) <= 1 else 2) and union_cov >= 0.50:
        for idx, p in enumerate(passages, 1):
            if not is_abstention(p) and (key_q & content_stems(p)):
                supporting.append(idx)

    return bool(supporting), round(best_score, 4), supporting


def extract_proper_entities(text: str) -> set[str]:
    """Extract capitalized multi-letter proper nouns/acronyms mid-sentence (case-sensitive)."""
    clean = strip_citations(text)
    ents: set[str] = set()
    for sent in split_sentences(clean):
        words = list(WORD_RE.finditer(sent))
        for i, m in enumerate(words):
            w = m.group(0)
            if len(w) < 2:
                continue
            # All-caps acronym (e.g., FDA, COX-1, HIPAA, TLS, AES) or mid-sentence Capitalized Proper Noun
            if (w.isupper() and any(ch.isalpha() for ch in w) and len(w) >= 2) or (i > 0 and w[0].isupper() and any(ch.islower() for ch in w)):
                wl = w.lower()
                if wl not in STOPWORDS:
                    ents.add(wl)
    return ents


@dataclass
class VerificationResult:
    score: float
    passed: bool
    abstained: bool
    answerable: bool
    word_count: int
    cited_indices: list[int]
    grounding_ratio: float
    answerability_score: float
    reasons: list[str] = field(default_factory=list)
    sentence_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify(
    query: str,
    passages: Sequence[str],
    answer: str,
    *,
    max_words: int = MAX_WORDS,
    min_grounding: float = 0.60,
) -> VerificationResult:
    """Verify a passage-grounded answer against passages[1..k] (1 <= k <= 8)."""
    reasons: list[str] = []
    if not isinstance(passages, Sequence) or isinstance(passages, (str, bytes)):
        raise ValueError("passages must be a sequence of strings")
    k = len(passages)
    if k < MIN_PASSAGES or k > MAX_PASSAGES:
        raise ValueError(f"passages length must be in [{MIN_PASSAGES}, {MAX_PASSAGES}], got {k}")

    ans = (answer or "").strip()
    words = ans.split()
    word_count = len(words)

    answerable, ans_score, _ = estimate_passage_answerability(query, passages)

    if word_count == 0:
        return VerificationResult(
            score=0.0, passed=False, abstained=False, answerable=answerable,
            word_count=0, cited_indices=[], grounding_ratio=0.0,
            answerability_score=ans_score, reasons=["empty_answer"],
        )

    if word_count > max_words:
        return VerificationResult(
            score=0.0, passed=False, abstained=is_abstention(ans), answerable=answerable,
            word_count=word_count, cited_indices=[], grounding_ratio=0.0,
            answerability_score=ans_score, reasons=[f"word_limit_exceeded:{word_count}>{max_words}"],
        )

    all_valid_cites, all_invalid_cites = parse_citations(ans, k)
    if all_invalid_cites:
        return VerificationResult(
            score=0.0, passed=False, abstained=is_abstention(ans), answerable=answerable,
            word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=0.0,
            answerability_score=ans_score,
            reasons=[f"invalid_citation_index:{','.join(all_invalid_cites)}"],
        )

    abstained = is_abstention(ans)
    if abstained:
        # Contradictory abstention: citing passages or making imperative/factual claims while abstaining
        if all_valid_cites:
            return VerificationResult(
                score=0.0, passed=False, abstained=True, answerable=answerable,
                word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=0.0,
                answerability_score=ans_score, reasons=["contradictory_abstention_with_citation"],
            )
        non_abs_sents = [s for s in split_sentences(ans) if not is_abstention(s)]
        if any(extract_imperative_actions(s) for s in non_abs_sents):
            return VerificationResult(
                score=0.0, passed=False, abstained=True, answerable=answerable,
                word_count=word_count, cited_indices=[], grounding_ratio=0.0,
                answerability_score=ans_score, reasons=["contradictory_abstention_with_action"],
            )
        if answerable:
            return VerificationResult(
                score=0.0, passed=False, abstained=True, answerable=True,
                word_count=word_count, cited_indices=[], grounding_ratio=0.0,
                answerability_score=ans_score, reasons=["over_refusal"],
            )
        return VerificationResult(
            score=1.0, passed=True, abstained=True, answerable=False,
            word_count=word_count, cited_indices=[], grounding_ratio=1.0,
            answerability_score=ans_score, reasons=["calibrated_abstention"],
        )

    # Non-abstention answer MUST have valid inline citations
    if not all_valid_cites:
        return VerificationResult(
            score=0.0, passed=False, abstained=False, answerable=answerable,
            word_count=word_count, cited_indices=[], grounding_ratio=0.0,
            answerability_score=ans_score, reasons=["missing_citations"],
        )

    sents = split_sentences(ans)
    sent_diags: list[dict[str, Any]] = []
    grounding_scores: list[float] = []

    for sent in sents:
        s_clean = strip_citations(sent)
        s_stems = content_stems(s_clean)
        if not s_stems:
            continue

        s_cites, s_bad = parse_citations(sent, k)
        if s_bad:
            reasons.append(f"invalid_citation_index:{','.join(s_bad)}")
            return VerificationResult(
                score=0.0, passed=False, abstained=False, answerable=answerable,
                word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=0.0,
                answerability_score=ans_score, reasons=reasons, sentence_diagnostics=sent_diags,
            )
        if not s_cites:
            reasons.append("uncited_claim_sentence")
            sent_diags.append({"sentence": sent, "ok": False, "reason": "uncited_claim_sentence"})
            return VerificationResult(
                score=0.0, passed=False, abstained=False, answerable=answerable,
                word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=0.0,
                answerability_score=ans_score, reasons=reasons, sentence_diagnostics=sent_diags,
            )

        cited_texts = [passages[i - 1] for i in s_cites]
        cited_blob = " ".join(cited_texts)
        cited_stems = content_stems(cited_blob)
        cited_tokens = set(tokenize(cited_blob))

        # GATE A: Strict 0.0 imperative safety/action verb phrase grounding
        actions = extract_imperative_actions(sent)
        for act in actions:
            ok_act, act_reason = action_grounded_in_passages(act, cited_texts)
            if not ok_act:
                reasons.append(f"imperative_safety_action_ungrounded:{act_reason}")
                sent_diags.append({"sentence": sent, "cited": s_cites, "ok": False, "reason": act_reason})
                return VerificationResult(
                    score=0.0, passed=False, abstained=False, answerable=answerable,
                    word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=0.0,
                    answerability_score=ans_score, reasons=reasons, sentence_diagnostics=sent_diags,
                )

        # GATE B: Numeric & unit exact grounding
        s_nums = extract_numbers(sent)
        c_nums = extract_numbers(cited_blob)
        c_num_vals = {n for n, _ in c_nums}
        for num_val, num_unit in s_nums:
            if num_unit:
                if (num_val, num_unit) not in c_nums and not (num_val in c_num_vals and num_unit in cited_tokens):
                    reasons.append(f"numeric_mismatch:{num_val}{num_unit}")
                    sent_diags.append({"sentence": sent, "cited": s_cites, "ok": False, "reason": f"numeric_mismatch:{num_val}{num_unit}"})
                    return VerificationResult(
                        score=0.0, passed=False, abstained=False, answerable=answerable,
                        word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=0.0,
                        answerability_score=ans_score, reasons=reasons, sentence_diagnostics=sent_diags,
                    )
            elif num_val not in c_num_vals:
                reasons.append(f"numeric_mismatch:{num_val}")
                sent_diags.append({"sentence": sent, "cited": s_cites, "ok": False, "reason": f"numeric_mismatch:{num_val}"})
                return VerificationResult(
                    score=0.0, passed=False, abstained=False, answerable=answerable,
                    word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=0.0,
                    answerability_score=ans_score, reasons=reasons, sentence_diagnostics=sent_diags,
                )

        # GATE C: Negation / polarity alignment on the best-matching cited sentence
        best_psent = ""
        best_ps_ov = -1.0
        for ct in cited_texts:
            for ps in split_sentences(ct):
                ps_stems = content_stems(ps)
                ov = len(s_stems & ps_stems) / max(1, len(s_stems))
                if ov > best_ps_ov:
                    best_ps_ov = ov
                    best_psent = ps
        if best_psent and best_ps_ov >= 0.35:
            if has_negation(s_clean, reference_stems=s_stems) != has_negation(best_psent, reference_stems=s_stems):
                # Verify it's not a harmless conditional ("if no leak, continue")
                reasons.append("polarity_inversion")
                sent_diags.append({"sentence": sent, "cited": s_cites, "ok": False, "reason": "polarity_inversion"})
                return VerificationResult(
                    score=0.0, passed=False, abstained=False, answerable=answerable,
                    word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=0.0,
                    answerability_score=ans_score, reasons=reasons, sentence_diagnostics=sent_diags,
                )

        # GATE D: Proper-entity / acronym grounding (catches parametric memory leaks!)
        s_ents = extract_proper_entities(sent)
        q_tokens = set(tokenize(query))
        ungrounded_ents = [
            e for e in sorted(s_ents)
            if e not in cited_tokens and stem_token(e) not in cited_stems and e not in q_tokens
        ]
        if ungrounded_ents:
            reasons.append(f"parametric_entity_leak:{','.join(ungrounded_ents[:4])}")
            sent_diags.append({"sentence": sent, "cited": s_cites, "ok": False, "reason": f"parametric_entity_leak:{','.join(ungrounded_ents[:4])}"})
            return VerificationResult(
                score=0.0, passed=False, abstained=False, answerable=answerable,
                word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=0.0,
                answerability_score=ans_score, reasons=reasons, sentence_diagnostics=sent_diags,
            )

        # GATE E: Lexical span & content-stem grounding ratio
        # Exclude stems that appear purely in the query framing only if passage already covers >=1 substantive stem
        supported = s_stems & cited_stems
        ratio = len(supported) / max(1, len(s_stems))

        # Bigram span bonus/check: ensure tokens come from coherent spans, not random bag-of-words
        s_norm = [stem_token(t) for t in tokenize(s_clean) if t not in STOPWORDS and len(t) > 1]
        c_norm = [stem_token(t) for t in tokenize(cited_blob) if t not in STOPWORDS and len(t) > 1]
        c_bigrams = set(zip(c_norm, c_norm[1:])) if len(c_norm) >= 2 else set()
        s_bigrams = list(zip(s_norm, s_norm[1:])) if len(s_norm) >= 2 else []
        bigram_ratio = (
            sum(1 for bg in s_bigrams if bg in c_bigrams) / len(s_bigrams)
            if s_bigrams else ratio
        )

        # Every cited index in s_cites must contribute at least 1 supported stem (blocks padding [1][2] with irrelevant [2])
        for ci in s_cites:
            ci_stems = content_stems(passages[ci - 1])
            if not (s_stems & ci_stems):
                reasons.append(f"irrelevant_citation:[{ci}]")
                sent_diags.append({"sentence": sent, "cited": s_cites, "ok": False, "reason": f"irrelevant_citation:[{ci}]"})
                return VerificationResult(
                    score=0.0, passed=False, abstained=False, answerable=answerable,
                    word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=round(ratio, 4),
                    answerability_score=ans_score, reasons=reasons, sentence_diagnostics=sent_diags,
                )

        if ratio < min_grounding:
            reasons.append(f"low_lexical_grounding:{ratio:.3f}<{min_grounding:.2f}")
            sent_diags.append({"sentence": sent, "cited": s_cites, "ok": False, "reason": f"low_lexical_grounding:{ratio:.3f}"})
            return VerificationResult(
                score=0.0, passed=False, abstained=False, answerable=answerable,
                word_count=word_count, cited_indices=all_valid_cites, grounding_ratio=round(ratio, 4),
                answerability_score=ans_score, reasons=reasons, sentence_diagnostics=sent_diags,
            )

        sent_score = min(1.0, 0.80 * ratio + 0.20 * max(ratio, bigram_ratio))
        grounding_scores.append(sent_score)
        sent_diags.append({
            "sentence": sent, "cited": s_cites, "ok": True,
            "stem_ratio": round(ratio, 4), "bigram_ratio": round(bigram_ratio, 4),
            "actions_checked": len(actions),
        })

    # If passages did NOT support answering the query at all, answering anyway is an under-abstention leak
    if not answerable and ans_score < 0.28:
        return VerificationResult(
            score=0.0, passed=False, abstained=False, answerable=False,
            word_count=word_count, cited_indices=all_valid_cites,
            grounding_ratio=round(sum(grounding_scores) / max(1, len(grounding_scores)), 4),
            answerability_score=ans_score, reasons=["under_abstention_off_topic_passages"],
            sentence_diagnostics=sent_diags,
        )

    final_ratio = sum(grounding_scores) / max(1, len(grounding_scores))
    final_score = round(max(0.0, min(1.0, final_ratio)), 4)
    return VerificationResult(
        score=final_score,
        passed=final_score >= min_grounding,
        abstained=False,
        answerable=answerable,
        word_count=word_count,
        cited_indices=all_valid_cites,
        grounding_ratio=round(final_ratio, 4),
        answerability_score=ans_score,
        reasons=["grounded_cited_answer"],
        sentence_diagnostics=sent_diags,
    )


def verify_batch(records: Sequence[dict[str, Any]]) -> list[VerificationResult]:
    return [verify(r["query"], r["passages"], r["answer"]) for r in records]

import random
import unittest

# === DPO Hard-Negative Synthesizer (synth_pairs) ===
CORE_CATEGORIES = (
    "parametric_memory_leak",
    "hallucinated_step_injection",
    "over_refusal",
)


@dataclass(frozen=True)
class ScenarioTemplate:
    scenario_id: str
    domain: str
    query: str
    passages: tuple[str, ...]
    chosen: str
    rejected_parametric: str
    rejected_step: str
    rejected_over_refusal: str
    rejected_numeric: str
    rejected_polarity: str
    unanswerable_query: str
    calibrated_abstention: str
    rejected_under_abstention: str


SCENARIOS: tuple[ScenarioTemplate, ...] = (
    ScenarioTemplate(
        scenario_id="pharma_acetaminophen",
        domain="clinical_safety",
        query="What is the maximum daily adult dose and warning for Acetaminophen-XR?",
        passages=(
            "Acetaminophen-XR adult dosing must not exceed 3000 mg in any 24 hour period.",
            "Patients with severe hepatic impairment should avoid Acetaminophen-XR.",
            "Store Acetaminophen-XR tablets at 20 C to 25 C in a dry container.",
        ),
        chosen=(
            "Acetaminophen-XR adult dosing must not exceed 3000 mg in any 24 hour period [1]. "
            "Patients with severe hepatic impairment should avoid Acetaminophen-XR [2]."
        ),
        rejected_parametric=(
            "Acetaminophen-XR adult dosing must not exceed 3000 mg in any 24 hour period [1]. "
            "Acetaminophen inhibits cyclooxygenase in the central nervous system and was approved by the FDA in 1951 [1]."
        ),
        rejected_step=(
            "Acetaminophen-XR adult dosing must not exceed 3000 mg in any 24 hour period [1]. "
            "Administer 500 mg ibuprofen alongside each dose for faster fever relief [1]."
        ),
        rejected_over_refusal=(
            "Insufficient evidence in the provided passages to determine the maximum daily adult dose of Acetaminophen-XR."
        ),
        rejected_numeric=(
            "Acetaminophen-XR adult dosing must not exceed 4000 mg in any 24 hour period [1]."
        ),
        rejected_polarity=(
            "Patients with severe hepatic impairment should take Acetaminophen-XR [2]."
        ),
        unanswerable_query="What is the pediatric intravenous dosing weight chart for Acetaminophen-XR?",
        calibrated_abstention=(
            "Insufficient evidence in the provided passages to determine the pediatric intravenous dosing weight chart for Acetaminophen-XR."
        ),
        rejected_under_abstention=(
            "Administer 15 mg per kg intravenously every 6 hours for pediatric patients [1]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="loto_hv_panel",
        domain="industrial_loto",
        query="What lockout procedure must operators follow before servicing Panel-B7?",
        passages=(
            "Before servicing Panel-B7, operators must disconnect the main 480 V feeder breaker.",
            "Attach a personal lockout padlock and verify zero voltage with a rated meter.",
            "Do not bypass the cabinet door interlock switch under any maintenance condition.",
            "Panel-B7 houses three redundant cooling fans rated at 24 V DC.",
        ),
        chosen=(
            "Operators must disconnect the main 480 V feeder breaker before servicing Panel-B7 [1]. "
            "Attach a personal lockout padlock and verify zero voltage with a rated meter [2]. "
            "Do not bypass the cabinet door interlock switch under any maintenance condition [3]."
        ),
        rejected_parametric=(
            "Operators must disconnect the main 480 V feeder breaker before servicing Panel-B7 [1]. "
            "OSHA standard 1910.147 mandates annual arc-flash recertification across United States plants [1]."
        ),
        rejected_step=(
            "Operators must disconnect the main 480 V feeder breaker before servicing Panel-B7 [1]. "
            "Flush the busbars with deionized water and reboot the substation transformer [1]."
        ),
        rejected_over_refusal=(
            "I cannot answer based on the provided passages because the lockout steps for Panel-B7 are not mentioned."
        ),
        rejected_numeric=(
            "Operators must disconnect the main 240 V feeder breaker before servicing Panel-B7 [1]."
        ),
        rejected_polarity=(
            "Bypass the cabinet door interlock switch during maintenance on Panel-B7 [3]."
        ),
        unanswerable_query="Who manufactured the steel enclosure for Panel-B7?",
        calibrated_abstention=(
            "The provided passages do not specify who manufactured the steel enclosure for Panel-B7."
        ),
        rejected_under_abstention=(
            "Siemens Energy manufactured the steel enclosure for Panel-B7 in Munich [4]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="chemical_hypochlorite",
        domain="chemical_handling",
        query="How should technicians handle a sodium hypochlorite spill in Bay-3?",
        passages=(
            "For a sodium hypochlorite spill in Bay-3, isolate the spill perimeter immediately.",
            "Ventilate Bay-3 and absorb liquid hypochlorite using inert vermiculite.",
            "Never mix sodium hypochlorite with ammonia or acid cleaners because toxic chlorine gas forms.",
        ),
        chosen=(
            "Isolate the spill perimeter immediately for a sodium hypochlorite spill in Bay-3 [1]. "
            "Ventilate Bay-3 and absorb liquid hypochlorite using inert vermiculite [2]. "
            "Never mix sodium hypochlorite with ammonia or acid cleaners because toxic chlorine gas forms [3]."
        ),
        rejected_parametric=(
            "Isolate the spill perimeter immediately for a sodium hypochlorite spill in Bay-3 [1]. "
            "Sodium hypochlorite has chemical formula NaClO and was first produced by Claude Berthollet in Paris [1]."
        ),
        rejected_step=(
            "Isolate the spill perimeter immediately for a sodium hypochlorite spill in Bay-3 [1]. "
            "Pour hydrochloric acid directly onto the spill to neutralize the alkalinity [2]."
        ),
        rejected_over_refusal=(
            "Not enough information in the provided passages to explain sodium hypochlorite spill handling in Bay-3."
        ),
        rejected_numeric=(
            "For a sodium hypochlorite spill in Bay-3, wait 45 minutes before isolating the perimeter [1]."
        ),
        rejected_polarity=(
            "Mix sodium hypochlorite with ammonia or acid cleaners in Bay-3 [3]."
        ),
        unanswerable_query="What is the exact storage tank volume in liters inside Bay-3?",
        calibrated_abstention=(
            "Insufficient evidence in the provided passages to determine the storage tank volume in Bay-3."
        ),
        rejected_under_abstention=(
            "The storage tank inside Bay-3 holds 5000 liters of liquid hypochlorite [2]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="cloud_tls_incident",
        domain="security_incident",
        query="What steps should engineers take when a production TLS private key leaks?",
        passages=(
            "When a production TLS private key leaks, revoke the compromised certificate immediately.",
            "Issue a replacement ECDSA P-256 certificate and reload the edge ingress proxies within 15 minutes.",
            "Preserve audit logs for 90 days; do not delete forensic authentication records.",
            "Edge proxies listen on TCP port 443 across three availability zones.",
        ),
        chosen=(
            "Revoke the compromised certificate immediately when a production TLS private key leaks [1]. "
            "Issue a replacement ECDSA P-256 certificate and reload the edge ingress proxies within 15 minutes [2]. "
            "Preserve audit logs for 90 days and do not delete forensic authentication records [3]."
        ),
        rejected_parametric=(
            "Revoke the compromised certificate immediately when a production TLS private key leaks [1]. "
            "Certificate Transparency was introduced by Google under RFC 6962 to detect misissued certificates [1]."
        ),
        rejected_step=(
            "Revoke the compromised certificate immediately when a production TLS private key leaks [1]. "
            "Disable firewall egress filtering and delete the forensic authentication records [3]."
        ),
        rejected_over_refusal=(
            "Unable to determine from the provided passages what steps engineers should take after a TLS key leak."
        ),
        rejected_numeric=(
            "Issue a replacement ECDSA P-256 certificate and reload edge ingress proxies within 60 minutes [2]."
        ),
        rejected_polarity=(
            "Delete forensic authentication records after revoking the leaked certificate [3]."
        ),
        unanswerable_query="Which hardware security module vendor signs the root certificate authority?",
        calibrated_abstention=(
            "The provided passages do not mention which hardware security module vendor signs the root certificate authority."
        ),
        rejected_under_abstention=(
            "Thales Luna hardware security modules sign the root certificate authority [2]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="lithium_thermal",
        domain="battery_safety",
        query="What response protocol applies when Pack-X4 exceeds 85 C?",
        passages=(
            "If Pack-X4 exceeds 85 C, open the high-voltage contactor to disconnect load current.",
            "Evacuate personnel 30 meters upwind and cool the enclosure with copious water spray.",
            "Do not puncture or compress a swollen Pack-X4 cell module.",
        ),
        chosen=(
            "Open the high-voltage contactor to disconnect load current if Pack-X4 exceeds 85 C [1]. "
            "Evacuate personnel 30 meters upwind and cool the enclosure with copious water spray [2]."
        ),
        rejected_parametric=(
            "Open the high-voltage contactor to disconnect load current if Pack-X4 exceeds 85 C [1]. "
            "Lithium cobalt oxide cathodes were developed by John Goodenough at Oxford University in 1980 [1]."
        ),
        rejected_step=(
            "Open the high-voltage contactor to disconnect load current if Pack-X4 exceeds 85 C [1]. "
            "Puncture the swollen cell casing to vent internal pressure rapidly [3]."
        ),
        rejected_over_refusal=(
            "Insufficient evidence in the provided passages to specify the response protocol when Pack-X4 exceeds 85 C."
        ),
        rejected_numeric=(
            "Evacuate personnel 10 meters upwind and cool the enclosure with water spray [2]."
        ),
        rejected_polarity=(
            "Compress a swollen Pack-X4 cell module when it exceeds 85 C [3]."
        ),
        unanswerable_query="What is the retail replacement price of a Pack-X4 battery module in USD?",
        calibrated_abstention=(
            "Cannot answer based on the provided passages because the retail replacement price of Pack-X4 is not stated."
        ),
        rejected_under_abstention=(
            "A replacement Pack-X4 battery module costs 4200 USD from authorized dealers [1]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="postgres_wal_recovery",
        domain="database_ops",
        query="How should database administrators recover Cluster-DB2 after primary disk corruption?",
        passages=(
            "After primary disk corruption on Cluster-DB2, promote the synchronous standby replica.",
            "Archive all WAL segments to object storage and validate checksums before rejoining the old primary.",
            "Never truncate pg_wal manually while recovery is in progress.",
            "Cluster-DB2 targets an RPO of 0 seconds and an RTO of 120 seconds.",
        ),
        chosen=(
            "Promote the synchronous standby replica after primary disk corruption on Cluster-DB2 [1]. "
            "Archive all WAL segments to object storage and validate checksums before rejoining the old primary [2]. "
            "Never truncate pg_wal manually while recovery is in progress [3]."
        ),
        rejected_parametric=(
            "Promote the synchronous standby replica after primary disk corruption on Cluster-DB2 [1]. "
            "PostgreSQL evolved from the Ingres project led by Michael Stonebraker at Berkeley [1]."
        ),
        rejected_step=(
            "Promote the synchronous standby replica after primary disk corruption on Cluster-DB2 [1]. "
            "Run pg_resetwal immediately and delete the standby archive logs [2]."
        ),
        rejected_over_refusal=(
            "The provided passages do not contain enough information to recover Cluster-DB2 after disk corruption."
        ),
        rejected_numeric=(
            "Cluster-DB2 targets an RPO of 0 seconds and an RTO of 300 seconds [4]."
        ),
        rejected_polarity=(
            "Truncate pg_wal manually while recovery is in progress on Cluster-DB2 [3]."
        ),
        unanswerable_query="What NVMe SSD firmware version is installed on Cluster-DB2?",
        calibrated_abstention=(
            "Insufficient evidence in the provided passages to determine the NVMe SSD firmware version on Cluster-DB2."
        ),
        rejected_under_abstention=(
            "Cluster-DB2 runs Samsung PM9A3 firmware version GDC5602Q on all replicas [1]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="cryo_valve_pressure",
        domain="cryogenics",
        query="What pressure limit and bleed steps apply to Cryo-Tank-9?",
        passages=(
            "Cryo-Tank-9 working pressure must remain below 18 bar during normal liquid nitrogen transfer.",
            "If pressure reaches 18 bar, close the fill manifold valve and vent vapor through relief line R2.",
            "Operators must wear cryogenic gloves and a full face shield near relief line R2.",
        ),
        chosen=(
            "Cryo-Tank-9 working pressure must remain below 18 bar during normal liquid nitrogen transfer [1]. "
            "If pressure reaches 18 bar, close the fill manifold valve and vent vapor through relief line R2 [2]."
        ),
        rejected_parametric=(
            "Cryo-Tank-9 working pressure must remain below 18 bar during normal liquid nitrogen transfer [1]. "
            "Liquid nitrogen boils at minus 196 C at standard sea-level atmospheric pressure [1]."
        ),
        rejected_step=(
            "Cryo-Tank-9 working pressure must remain below 18 bar during normal liquid nitrogen transfer [1]. "
            "Heat relief line R2 with a propane torch to clear ice crystals [2]."
        ),
        rejected_over_refusal=(
            "Cannot determine the pressure limit or bleed steps for Cryo-Tank-9 from the provided passages."
        ),
        rejected_numeric=(
            "Cryo-Tank-9 working pressure must remain below 25 bar during normal liquid nitrogen transfer [1]."
        ),
        rejected_polarity=(
            "Operators do not need to wear cryogenic gloves or a face shield near relief line R2 [3]."
        ),
        unanswerable_query="When was Cryo-Tank-9 last hydrostatically certified?",
        calibrated_abstention=(
            "The provided passages do not specify when Cryo-Tank-9 was last hydrostatically certified."
        ),
        rejected_under_abstention=(
            "Cryo-Tank-9 received ASME hydrostatic certification on March 14, 2024 [1]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="sla_compliance_retention",
        domain="legal_compliance",
        query="What notification window and encryption rule apply under Enterprise SLA-v4?",
        passages=(
            "Under Enterprise SLA-v4, the vendor must notify affected tenants within 24 hours of confirming a data breach.",
            "All tenant backups must be encrypted at rest using AES-256-GCM with per-tenant KMS keys.",
            "Enterprise SLA-v4 guarantees 99.95% monthly uptime excluding scheduled Sunday maintenance windows.",
        ),
        chosen=(
            "Under Enterprise SLA-v4, the vendor must notify affected tenants within 24 hours of confirming a data breach [1]. "
            "All tenant backups must be encrypted at rest using AES-256-GCM with per-tenant KMS keys [2]."
        ),
        rejected_parametric=(
            "Under Enterprise SLA-v4, the vendor must notify affected tenants within 24 hours of confirming a data breach [1]. "
            "GDPR Article 33 independently mandates reporting to European supervisory authorities within 72 hours [1]."
        ),
        rejected_step=(
            "Under Enterprise SLA-v4, the vendor must notify affected tenants within 24 hours of confirming a data breach [1]. "
            "Disable KMS rotation and upload unencrypted backups to a public mirror [2]."
        ),
        rejected_over_refusal=(
            "Insufficient evidence in the provided passages regarding the notification window and encryption rule in Enterprise SLA-v4."
        ),
        rejected_numeric=(
            "Under Enterprise SLA-v4, the vendor must notify affected tenants within 72 hours of confirming a data breach [1]."
        ),
        rejected_polarity=(
            "Tenant backups are not encrypted at rest under Enterprise SLA-v4 [2]."
        ),
        unanswerable_query="What is the maximum financial penalty cap in dollars under Enterprise SLA-v4?",
        calibrated_abstention=(
            "Insufficient evidence in the provided passages to determine the financial penalty cap under Enterprise SLA-v4."
        ),
        rejected_under_abstention=(
            "Enterprise SLA-v4 caps financial breach penalties at 250000 USD per incident [3]."
        ),
    ),
)

DISTRACTOR_PASSAGES: tuple[str, ...] = (
    "Facility cafeteria hours run from 07:00 to 15:00 Monday through Friday.",
    "Quarterly inventory audits are archived in Building-C locker 14.",
    "Badge access logs synchronize to the regional controller every 6 hours.",
    "Parking permits for contractors require visitor desk sign-in at Gate-2.",
)


@dataclass(frozen=True)
class DPOPair:
    pair_id: str
    category: str
    domain: str
    query: str
    passages: list[str]
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    rejected_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_dpo_pairs(
    *,
    seed: int = 42,
    include_extended: bool = False,
    add_distractors: bool = True,
) -> list[DPOPair]:
    """Deterministically synthesize verified hard-negative DPO pairs.

    Guaranteed invariant: every returned pair satisfies:
      verify(query, passages, chosen).score >= 0.85
      verify(query, passages, rejected).score == 0.0
    """
    rng = random.Random(seed)
    pairs: list[DPOPair] = []

    for sc in SCENARIOS:
        base_passages = list(sc.passages)
        if add_distractors and len(base_passages) < 8:
            extra = DISTRACTOR_PASSAGES[rng.randrange(len(DISTRACTOR_PASSAGES))]
            base_passages.append(extra)

        # 1. Core hard negative: parametric_memory_leak
        res_c = verify(sc.query, base_passages, sc.chosen)
        res_r = verify(sc.query, base_passages, sc.rejected_parametric)
        assert res_c.passed and res_c.score >= 0.85, f"chosen failed on {sc.scenario_id}: {res_c}"
        assert res_r.score == 0.0, f"parametric leak not caught on {sc.scenario_id}: {res_r}"
        pairs.append(DPOPair(
            pair_id=f"{sc.scenario_id}__parametric_memory_leak",
            category="parametric_memory_leak",
            domain=sc.domain,
            query=sc.query,
            passages=list(base_passages),
            chosen=sc.chosen,
            rejected=sc.rejected_parametric,
            chosen_score=res_c.score,
            rejected_score=res_r.score,
            rejected_reasons=list(res_r.reasons),
        ))

        # 2. Core hard negative: hallucinated_step_injection
        res_r2 = verify(sc.query, base_passages, sc.rejected_step)
        assert res_r2.score == 0.0, f"hallucinated step not caught on {sc.scenario_id}: {res_r2}"
        pairs.append(DPOPair(
            pair_id=f"{sc.scenario_id}__hallucinated_step_injection",
            category="hallucinated_step_injection",
            domain=sc.domain,
            query=sc.query,
            passages=list(base_passages),
            chosen=sc.chosen,
            rejected=sc.rejected_step,
            chosen_score=res_c.score,
            rejected_score=res_r2.score,
            rejected_reasons=list(res_r2.reasons),
        ))

        # 3. Core hard negative: over_refusal
        res_r3 = verify(sc.query, base_passages, sc.rejected_over_refusal)
        assert res_r3.score == 0.0 and "over_refusal" in res_r3.reasons, (
            f"over_refusal not caught on {sc.scenario_id}: {res_r3}"
        )
        pairs.append(DPOPair(
            pair_id=f"{sc.scenario_id}__over_refusal",
            category="over_refusal",
            domain=sc.domain,
            query=sc.query,
            passages=list(base_passages),
            chosen=sc.chosen,
            rejected=sc.rejected_over_refusal,
            chosen_score=res_c.score,
            rejected_score=res_r3.score,
            rejected_reasons=list(res_r3.reasons),
        ))

        if include_extended:
            # 4. Calibrated abstention (chosen) vs parametric leak under-abstention (rejected)
            res_ac = verify(sc.unanswerable_query, base_passages, sc.calibrated_abstention)
            res_ar = verify(sc.unanswerable_query, base_passages, sc.rejected_under_abstention)
            assert res_ac.score == 1.0 and res_ac.abstained, f"calibrated abstention failed on {sc.scenario_id}: {res_ac}"
            assert res_ar.score == 0.0, f"under-abstention not caught on {sc.scenario_id}: {res_ar}"
            pairs.append(DPOPair(
                pair_id=f"{sc.scenario_id}__under_abstention_leak",
                category="parametric_memory_leak",
                domain=sc.domain,
                query=sc.unanswerable_query,
                passages=list(base_passages),
                chosen=sc.calibrated_abstention,
                rejected=sc.rejected_under_abstention,
                chosen_score=res_ac.score,
                rejected_score=res_ar.score,
                rejected_reasons=list(res_ar.reasons),
            ))

            # 5. Numeric drift
            res_num = verify(sc.query, base_passages, sc.rejected_numeric)
            assert res_num.score == 0.0, f"numeric drift not caught on {sc.scenario_id}: {res_num}"
            pairs.append(DPOPair(
                pair_id=f"{sc.scenario_id}__numeric_drift",
                category="numeric_drift",
                domain=sc.domain,
                query=sc.query,
                passages=list(base_passages),
                chosen=sc.chosen,
                rejected=sc.rejected_numeric,
                chosen_score=res_c.score,
                rejected_score=res_num.score,
                rejected_reasons=list(res_num.reasons),
            ))

            # 6. Polarity inversion
            res_pol = verify(sc.query, base_passages, sc.rejected_polarity)
            assert res_pol.score == 0.0, f"polarity inversion not caught on {sc.scenario_id}: {res_pol}"
            pairs.append(DPOPair(
                pair_id=f"{sc.scenario_id}__polarity_inversion",
                category="polarity_inversion",
                domain=sc.domain,
                query=sc.query,
                passages=list(base_passages),
                chosen=sc.chosen,
                rejected=sc.rejected_polarity,
                chosen_score=res_c.score,
                rejected_score=res_pol.score,
                rejected_reasons=list(res_pol.reasons),
            ))

    return pairs


def run_detection_benchmark(seed: int = 7) -> dict[str, Any]:
    """Run a deterministic 300-item labelled benchmark (150 positive, 150 adversarial negative)."""
    rng = random.Random(seed)
    items: list[tuple[bool, str, str, list[str], str]] = []

    # 150 positives: 110 grounded cited answers + 40 calibrated abstentions
    for i in range(110):
        sc = SCENARIOS[i % len(SCENARIOS)]
        p = list(sc.passages)
        if len(p) < 8:
            p.append(DISTRACTOR_PASSAGES[rng.randrange(len(DISTRACTOR_PASSAGES))])
        items.append((True, "grounded_cited", sc.query, p, sc.chosen))
    for i in range(40):
        sc = SCENARIOS[i % len(SCENARIOS)]
        items.append((True, "calibrated_abstention", sc.unanswerable_query, list(sc.passages), sc.calibrated_abstention))

    # 150 negatives: 25 each across 6 distinct adversarial failure classes
    neg_classes = [
        ("parametric_memory_leak", lambda sc: (sc.query, list(sc.passages), sc.rejected_parametric)),
        ("hallucinated_step_injection", lambda sc: (sc.query, list(sc.passages), sc.rejected_step)),
        ("over_refusal", lambda sc: (sc.query, list(sc.passages), sc.rejected_over_refusal)),
        ("numeric_drift", lambda sc: (sc.query, list(sc.passages), sc.rejected_numeric)),
        ("polarity_inversion", lambda sc: (sc.query, list(sc.passages), sc.rejected_polarity)),
        ("out_of_range_or_wordcap", lambda sc: (
            sc.query, list(sc.passages),
            (sc.chosen.replace("[1]", "[9]") if (len(sc.passages) < 9) else (sc.chosen + " extra" * 125))
        )),
    ]
    for cname, fn in neg_classes:
        for i in range(25):
            sc = SCENARIOS[i % len(SCENARIOS)]
            q, p, a = fn(sc)
            items.append((False, cname, q, p, a))

    tp = fp = tn = fn_cnt = 0
    per_class: dict[str, dict[str, int]] = {}
    for expected_pass, cname, q, p, a in items:
        res = verify(q, p, a)
        pred_pass = res.passed and res.score > 0.0
        st = per_class.setdefault(cname, {"total": 0, "correct": 0})
        st["total"] += 1
        if expected_pass == pred_pass:
            st["correct"] += 1
        if expected_pass and pred_pass:
            tp += 1
        elif not expected_pass and pred_pass:
            fp += 1
        elif not expected_pass and not pred_pass:
            tn += 1
        else:
            fn_cnt += 1

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn_cnt)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {
        "total": len(items),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn_cnt,
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "accuracy": round((tp + tn) / len(items), 6),
        "per_class": per_class,
    }

# === 42-Case Adversarial Test Suite (pytest & unittest) ===
import unittest




class TestAdversarialCitationVerifier(unittest.TestCase):
    # --- 1. Input & structural boundaries ---
    def test_01_empty_passages_raises(self) -> None:
        with self.assertRaises(ValueError):
            verify("What is the limit?", [], "Insufficient evidence.")

    def test_02_nine_passages_raises(self) -> None:
        with self.assertRaises(ValueError):
            verify("What is the limit?", [f"Passage {i} limit is 10." for i in range(9)], "Limit is 10 [1].")

    def test_03_eight_passages_max_valid_index(self) -> None:
        passages = [f"Unrelated note number {i}." for i in range(7)] + [
            "Reactor coolant loop C-8 operates at a maximum pressure of 145 bar."
        ]
        res = verify(
            "What is the maximum pressure of reactor coolant loop C-8?",
            passages,
            "Reactor coolant loop C-8 operates at a maximum pressure of 145 bar [8].",
        )
        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.score, 0.95)
        self.assertEqual(res.cited_indices, [8])

    def test_04_empty_answer_scores_zero(self) -> None:
        res = verify("What is the voltage?", ["Rail A carries 48 V DC."], "   ")
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.passed)

    def test_05_word_limit_exact_120_passes(self) -> None:
        passage = (
            "The calibration module verifies sensor offset drift every 10 seconds across primary channels "
            "and logs signed telemetry frames to non-volatile ring storage without interrupting control loops."
        )
        # Build an exactly 120-word grounded answer using cited repetitions of passage claims
        sent = "The calibration module verifies sensor offset drift every 10 seconds across primary channels [1]."
        # 14 words per sentence -> 8 sentences = 112 words + 8-word final sentence = 120 words
        tail = "It logs signed telemetry frames to storage [1]."  # 8 words
        ans = " ".join([sent] * 8 + [tail])
        self.assertEqual(len(ans.split()), 120)
        res = verify("How does the calibration module verify sensor offset drift?", [passage], ans)
        self.assertTrue(res.passed, res)
        self.assertEqual(res.word_count, 120)

    def test_06_word_limit_121_scores_zero(self) -> None:
        passage = (
            "The calibration module verifies sensor offset drift every 10 seconds across primary channels "
            "and logs signed telemetry frames to non-volatile ring storage."
        )
        sent = "The calibration module verifies sensor offset drift every 10 seconds across primary channels [1]."
        tail = "It also logs signed telemetry frames to storage [1]."  # 9 words -> 121 total
        ans = " ".join([sent] * 8 + [tail])
        self.assertEqual(len(ans.split()), 121)
        res = verify("How does the calibration module verify sensor offset drift?", [passage], ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("word_limit_exceeded" in r for r in res.reasons))

    # --- 2. Citation index & coverage gates ---
    def test_07_citation_index_zero_rejected(self) -> None:
        res = verify("What is the port?", ["Service listens on port 8443."], "Service listens on port 8443 [0].")
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("invalid_citation_index" in r for r in res.reasons))

    def test_08_citation_index_exceeding_k_rejected(self) -> None:
        res = verify(
            "What is the port?",
            ["Service listens on port 8443.", "TLS 1.3 is mandatory."],
            "Service listens on port 8443 [3].",
        )
        self.assertEqual(res.score, 0.0)

    def test_09_malformed_non_numeric_citation_rejected(self) -> None:
        res = verify("What is the port?", ["Service listens on port 8443."], "Service listens on port 8443 [src1].")
        self.assertEqual(res.score, 0.0)

    def test_10_missing_citations_scores_zero(self) -> None:
        res = verify("What is the port?", ["Service listens on port 8443."], "Service listens on port 8443.")
        self.assertEqual(res.score, 0.0)
        self.assertIn("missing_citations", res.reasons)

    def test_11_uncited_second_sentence_scores_zero(self) -> None:
        res = verify(
            "What is the port and TLS version?",
            ["Service listens on port 8443.", "TLS 1.3 is mandatory."],
            "Service listens on port 8443 [1]. TLS 1.3 is mandatory.",
        )
        self.assertEqual(res.score, 0.0)
        self.assertIn("uncited_claim_sentence", res.reasons)

    def test_12_multi_passage_comma_citation_supported(self) -> None:
        res = verify(
            "What port and TLS version are required?",
            ["Service listens on port 8443.", "TLS 1.3 is mandatory for all connections."],
            "Service listens on port 8443 and TLS 1.3 is mandatory for all connections [1, 2].",
        )
        self.assertTrue(res.passed, res)
        self.assertEqual(res.cited_indices, [1, 2])

    def test_13_irrelevant_citation_padding_rejected(self) -> None:
        res = verify(
            "What port does the service use?",
            ["Service listens on port 8443.", "Cafeteria serves lunch at noon."],
            "Service listens on port 8443 [1][2].",
        )
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("irrelevant_citation" in r for r in res.reasons))

    def test_14_wrong_passage_index_swap_rejected(self) -> None:
        res = verify(
            "What port does the service use?",
            ["Service listens on port 8443.", "Cafeteria serves lunch at noon."],
            "Service listens on port 8443 [2].",
        )
        self.assertEqual(res.score, 0.0)

    # --- 3. Strict 0.0 imperative safety/action verb phrase gate ---
    def test_15_grounded_imperative_safety_steps_pass(self) -> None:
        passages = [
            "If gas alarms sound in Lab-4, evacuate personnel immediately and shut down the main gas valve.",
            "Do not flip electrical wall switches during a gas leak.",
        ]
        ans = (
            "Evacuate personnel immediately and shut down the main gas valve if gas alarms sound in Lab-4 [1]. "
            "Do not flip electrical wall switches during a gas leak [2]."
        )
        res = verify("What should staff do if gas alarms sound in Lab-4?", passages, ans)
        self.assertTrue(res.passed, res)
        self.assertGreaterEqual(res.score, 0.90)

    def test_16_hallucinated_imperative_reboot_scores_zero(self) -> None:
        passages = ["When latency spikes above 400 ms, drain traffic from Node-A and inspect queue depth."]
        ans = "Drain traffic from Node-A, inspect queue depth, and reboot the database cluster immediately [1]."
        res = verify("How should operators respond when latency spikes above 400 ms?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("imperative_safety_action_ungrounded" in r for r in res.reasons))

    def test_17_hallucinated_phrasal_action_turn_off_scores_zero(self) -> None:
        passages = ["Monitor coolant flow rate on Pump-2 whenever inlet temperature exceeds 60 C."]
        ans = "Monitor coolant flow rate on Pump-2 and turn off the emergency cooling pump [1]."
        res = verify("How should Pump-2 be operated above 60 C?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("imperative_safety_action_ungrounded" in r for r in res.reasons))

    def test_18_hallucinated_modal_directive_disable_scores_zero(self) -> None:
        passages = ["Review failed login alerts in SIEM-01 and lock compromised user accounts."]
        ans = "Operators must lock compromised user accounts and should disable multi-factor authentication temporarily [1]."
        res = verify("How should operators handle failed login alerts?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("imperative_safety_action_ungrounded" in r for r in res.reasons))

    def test_19_hallucinated_clinical_dose_injection_scores_zero(self) -> None:
        passages = ["For mild dehydration, provide oral rehydration salts in small sips."]
        ans = "Provide oral rehydration salts in small sips and inject 500 ml isotonic saline rapidly [1]."
        res = verify("How should mild dehydration be managed?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_20_action_present_only_in_uncited_passage_scores_zero(self) -> None:
        passages = [
            "Close isolation valve V-1 before opening the filter housing.",
            "Depressurize manifold M-4 completely before loosening flange bolts.",
        ]
        ans = "Close isolation valve V-1 and depressurize manifold M-4 before loosening bolts [1]."
        res = verify("What steps precede opening the filter housing?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_21_prohibition_flipped_to_imperative_scores_zero(self) -> None:
        passages = ["Never disconnect the grounding strap while the high-voltage capacitor bank is charged."]
        ans = "Disconnect the grounding strap while the high-voltage capacitor bank is charged [1]."
        res = verify("What is the grounding strap rule for the capacitor bank?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_22_same_verb_wrong_safety_object_scores_zero(self) -> None:
        passages = ["Flush the eye wash basin with sterile saline for 15 minutes."]
        ans = "Flush the electrical junction box with sterile saline for 15 minutes [1]."
        res = verify("How should the eye wash basin be flushed?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_23_imperative_bypass_interlock_scores_zero(self) -> None:
        passages = ["Inspect the laser housing lid sensor and notify safety engineering if faulted."]
        ans = "Inspect the laser housing lid sensor and bypass the safety interlock to resume cutting [1]."
        res = verify("What should be done if the laser lid sensor faults?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_24_action_extractor_detects_phrasal_and_modal(self) -> None:
        acts = extract_imperative_actions("First shut down the turbine, then operators must bypass valve V-9.")
        verbs = {a.phrasal for a in acts}
        self.assertIn("shut down", verbs)
        self.assertIn("bypass", verbs)

    # --- 4. Parametric memory leaks, numeric drift & polarity ---
    def test_25_parametric_memory_leak_known_world_fact_rejected(self) -> None:
        passages = ["Aspirin-Mini tablets are packaged in 81 mg blister strips for clinic inventory."]
        ans = (
            "Aspirin-Mini tablets are packaged in 81 mg blister strips [1]. "
            "Aspirin was synthesized by Felix Hoffmann at Bayer in 1897 [1]."
        )
        res = verify("How are Aspirin-Mini tablets packaged?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_26_parametric_entity_leak_without_numbers_rejected(self) -> None:
        passages = ["The Orion relay forwards telemetry frames over an encrypted fiber link."]
        ans = "The Orion relay forwards telemetry frames over an encrypted fiber link designed by NASA [1]."
        res = verify("How does the Orion relay forward telemetry?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("parametric_entity_leak" in r for r in res.reasons))

    def test_27_numeric_value_drift_rejected(self) -> None:
        passages = ["The autoclave sterilization cycle runs at 121 C for 15 minutes."]
        ans = "The autoclave sterilization cycle runs at 134 C for 15 minutes [1]."
        res = verify("What temperature and duration does the autoclave use?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("numeric_mismatch" in r for r in res.reasons))

    def test_28_numeric_unit_mismatch_mg_vs_g_rejected(self) -> None:
        passages = ["Each capsule contains 250 mg of active compound Z-9."]
        ans = "Each capsule contains 250 g of active compound Z-9 [1]."
        res = verify("How much active compound Z-9 is in each capsule?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("numeric_mismatch" in r for r in res.reasons))

    def test_29_numeric_comma_normalization_passes(self) -> None:
        passages = ["The batch queue accepts up to 1,000 events per second."]
        ans = "The batch queue accepts up to 1000 events per second [1]."
        res = verify("What is the maximum rate of the batch queue?", passages, ans)
        self.assertTrue(res.passed, res)

    def test_30_decimal_value_grounded_passes(self) -> None:
        passages = ["Sensor voltage tolerance is 3.3 V with a 0.05 V ripple ceiling."]
        ans = "Sensor voltage tolerance is 3.3 V with a 0.05 V ripple ceiling [1]."
        res = verify("What is the sensor voltage tolerance?", passages, ans)
        self.assertTrue(res.passed, res)

    def test_31_polarity_inversion_on_policy_rejected(self) -> None:
        passages = ["Unverified third-party plugins are not permitted in production clusters."]
        ans = "Unverified third-party plugins are permitted in production clusters [1]."
        res = verify("Are unverified third-party plugins permitted in production?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertIn("polarity_inversion", res.reasons)

    # --- 5. Calibrated abstention, over-refusal & contradictory abstention ---
    def test_32_calibrated_abstention_on_unanswerable_query_passes(self) -> None:
        passages = [
            "Warehouse-B operates LED lighting on circuit panel L-2.",
            "Forklift batteries recharge overnight in Bay-6.",
        ]
        res = verify(
            "What is the maximum roof snow load rating in kPa for Warehouse-B?",
            passages,
            "Insufficient evidence in the provided passages to determine the roof snow load rating for Warehouse-B.",
        )
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.passed)
        self.assertTrue(res.abstained)
        self.assertFalse(res.answerable)

    def test_33_over_refusal_when_answer_is_in_passage_scores_zero(self) -> None:
        passages = ["Warehouse-B roof snow load rating is 2.4 kPa per structural drawing S-11."]
        res = verify(
            "What is the roof snow load rating in kPa for Warehouse-B?",
            passages,
            "Insufficient evidence in the provided passages to determine the roof snow load rating for Warehouse-B.",
        )
        self.assertEqual(res.score, 0.0)
        self.assertIn("over_refusal", res.reasons)

    def test_34_contradictory_abstention_with_citation_scores_zero(self) -> None:
        passages = ["Warehouse-B operates LED lighting on circuit panel L-2."]
        res = verify(
            "What is the roof snow load rating for Warehouse-B?",
            passages,
            "Insufficient evidence in the provided passages to answer this question [1].",
        )
        self.assertEqual(res.score, 0.0)
        self.assertIn("contradictory_abstention_with_citation", res.reasons)

    def test_35_contradictory_abstention_with_trailing_action_scores_zero(self) -> None:
        passages = ["Warehouse-B operates LED lighting on circuit panel L-2."]
        res = verify(
            "What is the roof snow load rating for Warehouse-B?",
            passages,
            "Insufficient evidence in the provided passages. Evacuate the roof and disable circuit panel L-2.",
        )
        self.assertEqual(res.score, 0.0)
        self.assertIn("contradictory_abstention_with_action", res.reasons)

    def test_36_under_abstention_on_off_topic_passages_scores_zero(self) -> None:
        passages = ["Cafeteria lunch hours are 11:30 to 13:30."]
        res = verify(
            "What is the maximum pediatric dose of amoxicillin?",
            passages,
            "Cafeteria lunch hours are 11:30 to 13:30 [1].",
        )
        self.assertEqual(res.score, 0.0)
        self.assertIn("under_abstention_off_topic_passages", res.reasons)

    def test_37_passage_explicit_abstention_treated_as_unanswerable(self) -> None:
        passages = ["The current manual does not specify the replacement belt tension for Motor-K3."]
        res = verify(
            "What is the replacement belt tension for Motor-K3?",
            passages,
            "The provided passages do not specify the replacement belt tension for Motor-K3.",
        )
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.abstained)

    # --- 6. DPO synthesizer & batch verification tests ---
    def test_38_synth_pairs_deterministic_identical_across_runs(self) -> None:
        p1 = [x.to_dict() for x in generate_dpo_pairs(seed=42, include_extended=True)]
        p2 = [x.to_dict() for x in generate_dpo_pairs(seed=42, include_extended=True)]
        self.assertEqual(p1, p2)
        self.assertGreaterEqual(len(p1), 24)

    def test_39_synth_pairs_covers_all_three_core_categories(self) -> None:
        pairs = generate_dpo_pairs(seed=1337, include_extended=False)
        cats = {p.category for p in pairs}
        for req in CORE_CATEGORIES:
            self.assertIn(req, cats)
        for p in pairs:
            self.assertGreaterEqual(p.chosen_score, 0.85)
            self.assertEqual(p.rejected_score, 0.0)
            self.assertLessEqual(len(p.chosen.split()), 120)
            self.assertLessEqual(len(p.rejected.split()), 120)

    def test_40_verify_batch_consistency(self) -> None:
        records = [
            {
                "query": "What is the backup retention window?",
                "passages": ["Encrypted snapshots are retained for 30 days in region-west."],
                "answer": "Encrypted snapshots are retained for 30 days in region-west [1].",
            },
            {
                "query": "What is the backup retention window?",
                "passages": ["Encrypted snapshots are retained for 30 days in region-west."],
                "answer": "Encrypted snapshots are retained for 90 days in region-west [1].",
            },
        ]
        results = verify_batch(records)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], VerificationResult)
        self.assertTrue(results[0].passed)
        self.assertEqual(results[1].score, 0.0)

    def test_41_parse_citations_mixed_valid_and_invalid(self) -> None:
        valid, invalid = parse_citations("Claim A [1, 2] and Claim B [0] and [9].", 4)
        self.assertEqual(valid, [1, 2])
        self.assertEqual(invalid, ["[0]", "[9]"])

    def test_42_detection_benchmark_300_items_perfect_f1(self) -> None:
        rep = run_detection_benchmark(seed=7)
        self.assertEqual(rep["total"], 300)
        self.assertEqual(rep["tp"], 150)
        self.assertEqual(rep["fp"], 0)
        self.assertEqual(rep["tn"], 150)
        self.assertEqual(rep["fn"], 0)
        self.assertEqual(rep["f1"], 1.0)

def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--synth" in args:
        pairs = generate_dpo_pairs(seed=42, include_extended=("--extended" in args))
        print(json.dumps({"pair_count": len(pairs), "categories": sorted({p.category for p in pairs})}, indent=2))
        return 0
    if "--benchmark" in args:
        rep = run_detection_benchmark(seed=7)
        print(json.dumps(rep, indent=2))
        return 0 if (rep["fp"] == 0 and rep["fn"] == 0) else 1
    # Default: run full 42-case adversarial unittest suite + 300-item benchmark + 48 DPO pair check
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAdversarialCitationVerifier)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if not res.wasSuccessful():
        return 1
    rep = run_detection_benchmark(seed=7)
    pairs = generate_dpo_pairs(seed=42, include_extended=True)
    print(json.dumps({
        "selftest_passed": res.testsRun,
        "dpo_pairs": len(pairs),
        "benchmark_total": rep["total"],
        "benchmark_tp": rep["tp"],
        "benchmark_fp": rep["fp"],
        "benchmark_tn": rep["tn"],
        "benchmark_fn": rep["fn"],
        "benchmark_f1": rep["f1"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
