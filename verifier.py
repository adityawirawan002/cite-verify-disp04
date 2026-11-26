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


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--stdin":
        payload = json.loads(sys.stdin.read())
        if isinstance(payload, list):
            out = [r.to_dict() for r in verify_batch(payload)]
        else:
            out = verify(payload["query"], payload["passages"], payload["answer"]).to_dict()
        print(json.dumps(out, indent=2))
        return 0
    # Quick built-in smoke check
    res = verify(
        "What is the maximum operating temperature of the lithium pack?",
        [
            "The Pack-X4 lithium module has a maximum continuous operating temperature of 55 C.",
            "Do not submerge the Pack-X4 module in water.",
        ],
        "The Pack-X4 lithium module has a maximum continuous operating temperature of 55 C [1].",
    )
    print(json.dumps(res.to_dict(), indent=2))
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
