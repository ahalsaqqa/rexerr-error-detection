"""Rule-based error injection for an external test set with a different error generator.

ReXErr errors were written by GPT-4o. To test whether detectors rely on how that model writes, each original test report
receives exactly one error produced by a deterministic rule, so the label is exact and the wording is not GPT-4o's.
Each rule targets one native ReXErr type; a report gets a type only if the rule applies to it.
"""
import re

import numpy as np

SIDE = {"left": "right", "right": "left"}
SEVERITY = {"mild": "severe", "mildly": "severely", "moderate": "mild", "moderately": "mildly", "severe": "mild",
            "severely": "mildly", "small": "large", "large": "small", "minimal": "marked", "trace": "large"}
HOMOPHONE = {"pleural": "plural", "there": "their", "site": "sight", "seen": "scene", "base": "bass",
             "bases": "basses", "right": "write", "new": "knew", "two": "too"}
DEVICES = ["A right internal jugular central venous catheter terminates in the mid SVC.",
           "An endotracheal tube terminates 4 cm above the carina.",
           "A nasogastric tube courses below the diaphragm with the tip in the stomach.",
           "A left-sided pacemaker with leads in the right atrium and right ventricle is present.",
           "A right PICC line terminates in the lower SVC."]
NUMBER_UNIT = re.compile(r"\b(\d+(?:\.\d+)?)(\s?)(cm|mm)\b")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _match_case(src, repl):
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _swap_word(text, mapping, rng):
    hits = [m for m in re.finditer(r"\b(" + "|".join(mapping) + r")\b", text, flags=re.I)]
    if not hits:
        return None
    m = hits[rng.integers(len(hits))]
    return text[:m.start()] + _match_case(m.group(0), mapping[m.group(0).lower()]) + text[m.end():]


def change_location(text, rng):
    return _swap_word(text, SIDE, rng)


def change_severity(text, rng):
    return _swap_word(text, SEVERITY, rng)


def change_to_homophone(text, rng):
    return _swap_word(text, HOMOPHONE, rng)


def change_measurement(text, rng):
    hits = list(NUMBER_UNIT.finditer(text))
    if not hits:
        return None
    m = hits[rng.integers(len(hits))]
    value = float(m.group(1))
    new = value * 2 if value < 1 or rng.random() < 0.5 else value / 2
    s = f"{new:.1f}".rstrip("0").rstrip(".")
    return text[:m.start()] + s + m.group(2) + m.group(3) + text[m.end():]


def false_negation(text, rng):
    """Negate an affirmative statement ("There is X" -> "There is no X")."""
    hits = [m for m in re.finditer(r"\b(There (?:is|are)) (?:(?:a|an) )?(?!no\b|not\b|none\b)", text)]
    if not hits:
        return None
    m = hits[rng.integers(len(hits))]
    return text[:m.start()] + m.group(1) + " no " + text[m.end():]


def false_prediction(text, rng):
    """Turn a negated finding into an affirmed one ("There is no X" -> "There is X", "No X." -> "X.")."""
    hits = [m for m in re.finditer(r"\b(There (?:is|are)) no ", text)]
    hits += [m for m in re.finditer(r"(?:^|(?<=[.!?] ))No (?=[a-z])", text)]
    if not hits:
        return None
    m = hits[rng.integers(len(hits))]
    if m.group(0).startswith("There"):
        return text[:m.start()] + m.group(1) + " " + text[m.end():]
    rest = text[m.end():]
    return text[:m.start()] + rest[:1].upper() + rest[1:]


def add_typo(text, rng):
    words = [m for m in re.finditer(r"\b[A-Za-z]{6,}\b", text)]
    if not words:
        return None
    m = words[rng.integers(len(words))]
    w = m.group(0)
    i = int(rng.integers(1, len(w) - 2))
    typo = w[:i] + w[i + 1] + w[i] + w[i + 2:] if rng.random() < 0.5 else w[:i] + w[i + 1:]
    if typo == w:  # swapping a doubled letter changes nothing; drop it instead
        typo = w[:i] + w[i + 1:]
    return text[:m.start()] + typo + text[m.end():]


def add_repetitions(text, rng):
    sents = SENTENCE.split(text)
    if len(sents) < 2:
        return None
    i = int(rng.integers(len(sents)))
    return " ".join(sents[:i + 1] + [sents[i]] + sents[i + 1:])


def add_medical_device(text, rng):
    sents = SENTENCE.split(text)
    i = int(rng.integers(len(sents) + 1))
    return " ".join(sents[:i] + [DEVICES[rng.integers(len(DEVICES))]] + sents[i:])


RULES = {"change_location": change_location, "change_severity": change_severity,
         "change_to_homophone": change_to_homophone, "change_measurement": change_measurement,
         "false_negation": false_negation, "false_prediction": false_prediction, "add_typo": add_typo,
         "add_repetitions": add_repetitions, "add_medical_device": add_medical_device}


def apply(text, name, seed):
    """Apply rule `name` to `text`; returns the new text, or None when the rule does not apply."""
    new = RULES[name](text, np.random.default_rng(seed))
    return new if new is not None and new != text else None


def build(texts, seed=0):
    """Give each report one error, choosing among the applicable rules the type used least so far.

    Returns a list of (new_text, type), with (None, None) for reports to which no rule applies.
    """
    rng = np.random.default_rng(seed)
    counts = {name: 0 for name in RULES}
    out = []
    for i, text in enumerate(texts):
        options = {name: apply(text, name, seed * 100003 + i) for name in RULES}
        options = {k: v for k, v in options.items() if v is not None}
        if not options:
            out.append((None, None))
            continue
        low = min(counts[k] for k in options)
        name = sorted(k for k in options if counts[k] == low)[int(rng.integers(sum(counts[k] == low for k in options)))]
        counts[name] += 1
        out.append((options[name], name))
    return out
