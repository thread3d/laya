#!/usr/bin/env python3
"""Generate the parity fixtures `laya-java` asserts against, by running the real `laya`.

Nothing in `laya-java` may hand-write an expectation that this script can produce. A port's
tokenizer, its sequence budgets and its decode arithmetic all drift silently, and a hand-typed
expectation drifts with them; a generated one cannot. `sdk/typescript/scripts/sync_presets.py`
sets the same precedent for the TypeScript client's presets, including the `--check` gate CI runs.

    gen_fixtures.py             # write laya-java/fixtures/*.json
    gen_fixtures.py --check     # exit 1 if any committed fixture would change

`--check` is what CI runs: it regenerates into memory and diffs, so a change to `laya/` that moves
the contract fails the Java build instead of being discovered by a user.

Phase 0 families (no model weights needed -- these are the tables and the text the model is shown):

  lang_tables.json    every table `laya/lang.py` routes on, and every threshold constant
  presets.json        the preset question dicts, verbatim, as the model receives them
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.normpath(os.path.join(HERE, "..", "fixtures"))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _sorted_set(values):
    """A set as a sorted list: a fixture has to be byte-stable across runs and interpreters."""
    return sorted(values)


def lang_tables():
    """Every table and threshold `laya.lang` decides a route with.

    Read from the imported module, never parsed out of the source: `_SHARED_WORDS`,
    `_EN_ONLY_WORDS` and `_EN_COLLISION_WORDS` are *derived* at import time from `_STOP`, so
    parsing the file would miss the derivation and a port built from it would be subtly wrong.
    """
    from laya import lang

    return {
        "script_ranges": [
            # (name, [(lo, hi), ...]) with the bounds as ints -- a port needs the numbers, not
            # whatever repr the host language gives a range object.
            {"name": name, "ranges": [[int(lo), int(hi)] for lo, hi in ranges]}
            for name, ranges in lang._SCRIPT_RANGES
        ],
        "stop_words": {code: _sorted_set(words) for code, words in sorted(lang._STOP.items())},
        "short_swedish_words": _sorted_set(lang._SHORT_SWEDISH_WORDS),
        "non_en_diacritics": _sorted_set(lang._NON_EN_DIACRITICS),
        "shared_words": _sorted_set(lang._SHARED_WORDS),
        "nordic_overlap_words": _sorted_set(lang._NORDIC_OVERLAP_WORDS),
        "en_only_words": _sorted_set(lang._EN_ONLY_WORDS),
        "en_collision_words": _sorted_set(lang._EN_COLLISION_WORDS),
        "thresholds": {
            name: getattr(lang, name)
            for name in sorted(n for n in dir(lang)
                               if n.isupper() and isinstance(getattr(lang, n), (int, float))
                               and not isinstance(getattr(lang, n), bool))
        },
        "regexes": {
            # The PATTERN TEXT, not a compiled object: a port must translate these deliberately
            # (Python `\w` under `re.UNICODE` is not JavaScript's `\w`, and neither is Java's).
            name: getattr(lang, name).pattern
            for name in ("_WORD", "_IDENTIFIER", "_CODE_LINE", "_JOINED", "_LETTER_RUN")
            if hasattr(lang, name) and hasattr(getattr(lang, name), "pattern")
        },
    }


def presets():
    """The preset question dicts exactly as a caller receives them, and as the model is shown them.

    `laya.presets` builds these; one changed word is a different question and therefore a
    different answer, so a port must generate them rather than retype them.
    """
    from laya import presets as mod

    out = {}
    for name in sorted(n for n in dir(mod) if n.endswith("_questions") or n.endswith("Questions")):
        value = getattr(mod, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, dict):
            out[name] = value
    return out


# A corpus chosen to break a tokenizer port, not to flatter it. Each entry is (id, text); the id
# is the join key between this fixture and the Java test, so it must stay stable.
TOKENIZER_CORPUS = [
    ("empty", ""),
    ("space", " "),
    ("ascii-prose", "We were billed twice for March and want a refund today."),
    ("ascii-punct", "!!! ??? ... --- *** ((())) [[]] {{}} <<>> ~~~ ///"),
    ("digits", "0 1 42 007 3.14159 1,000,000 -17 1e9 0x1F 2026-10-04"),
    ("leading-space", "   leading spaces"),
    ("trailing-space", "trailing spaces   "),
    ("inner-runs", "a  b\t\tc\n\nd\r\ne   f"),
    ("newlines-only", "\n\n\n"),
    ("cjk-zh", "我们三月份被重复收费了两次，请今天退还重复的金额。"),
    ("cjk-ja", "三月に二重請求されました。至急返金してください。"),
    ("cjk-ko", "3월에 두 번 청구되었습니다. 환불해 주세요."),
    ("arabic", "لقد تم محاسبتنا مرتين في شهر مارس، يرجى رد المبلغ المكرر اليوم."),
    ("hebrew", "חויבנו פעמיים במרץ, אנא החזירו את הסכום הכפול."),
    ("devanagari", "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया राशि वापस करें।"),
    ("thai", "เราถูกเรียกเก็บเงินสองครั้งในเดือนมีนาคม"),
    ("cyrillic", "С нас дважды списали оплату в марте, верните деньги."),
    ("greek", "Μας χρέωσαν δύο φορές τον Μάρτιο, θέλουμε επιστροφή."),
    ("mixed-scripts", "Refund 退款 استرداد वापसी now!"),
    # NFC matters: english normalises NFC, multilingual does not. The same grapheme, composed and
    # decomposed, must tokenize the way the real tokenizer says -- not the way a port assumes.
    ("nfc-composed", "caf\u00e9 na\u00efve \u00fcber"),
    ("nfd-decomposed", "cafe\u0301 nai\u0308ve u\u0308ber"),
    ("combining-stack", "a\u0301\u0302\u0303\u0304\u0305"),
    ("emoji", "\U0001f600 \U0001f389 \u2764\ufe0f"),
    ("emoji-zwj", "\U0001f468\u200d\U0001f469\u200d\U0001f466 \U0001f3f3\ufe0f\u200d\U0001f308"),
    ("emoji-skin-tone", "\U0001f44d\U0001f3fd \U0001f64f\U0001f3ff"),
    # U+2581 is the Metaspace replacement itself: a port that substitutes spaces naively will
    # collide with a caller who sent this character on purpose.
    ("metaspace-char", "price\u2581list\u2581here"),
    # Characters absent from the multilingual vocabulary, so `byte_fallback` must fire and emit
    # `<0xNN>` tokens. The curated corpus above contains NOT ONE such character, and a port whose
    # fallback was disabled outright still passed every entry of it while diverging on 25% of a
    # 30,000-string random sweep. The fixture now carries the case the sweep found.
    ("byte-fallback-rare-scripts", "\U00010A00 \u0CF1 \U00016FE0 \U0001E900"),
    ("byte-fallback-in-prose", "refund \U00010A00 now please"),
    ("byte-fallback-adjacent", "\U00010A00\U00010A00\u0CF1"),
    ("zero-width", "a\u200bb\u200cc\u200dd\ufeffe"),
    ("rtl-marks", "a\u202eb\u202cc \u200f\u200e"),
    ("mask-literal-angle", "please <mask> this and <pad> that"),
    ("mask-literal-square", "please [MASK] this and [PAD] that"),
    ("unspaced-long", "a" * 2000),
    ("unspaced-cjk-long", "中" * 1000),
    ("repeated-word", "refund " * 400),
    ("one-char-lines", "\n".join("a" * 500)),
    ("url-ish", "see https://example.com/a/b?c=d&e=f#g and mail to a.b+c@example.co.uk"),
    ("code-ish", "if (x == 1) { return y[0].z(); } // comment"),
    ("tabs-json", '{"a": [1, 2], "b": {"c": null}}'),
]

# Option text long enough that `build_head`'s `max_length=48` truncation at the tokenizer bites.
LONG_OPTION = ("a refund of the duplicate charge together with written confirmation that the "
               "payment method on file has been removed and will not be charged again under any "
               "circumstances whatsoever, including renewals")


def tokenizer_ids():
    """Exact token ids per checkpoint for a corpus designed to break a port.

    Ids, not answers: a tokenizer divergence that changes an answer is almost impossible to
    localise from the answer, and trivial from the ids. Each checkpoint records what it IS as well
    -- model type, pre-tokenizer, normaliser, vocab and merge counts, special ids -- so a fixture
    generated against a different checkpoint cannot be mistaken for a matching one.
    """
    import os

    from laya.common import encode_text

    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         "/Users/nombauser/Documents/nomba-dev/laya-e2e/checkpoints")
    out = {}
    for name in ("english", "multilingual"):
        root = os.path.join(rig, name)
        if not os.path.isdir(root):
            out[name] = {"skipped": "checkpoint not present at %s" % root}
            continue
        # Resolve the tokenizer directory EXACTLY as laya resolves it at run time:
        # `os.path.join(model_dir, "tokenizer")`, falling back to the configured encoder
        # (`onnx_agent.py:156`, `agent.py:235`). Searching for the shortest path containing a
        # `tokenizer.json` is not the same thing and silently picked the wrong directory: the
        # multilingual checkpoint ships `tokenizer.json` at BOTH its root and its `tokenizer/`
        # subdirectory -- byte-identical, same sha256 -- but only the subdirectory carries
        # `tokenizer_config.json`. Loading the root therefore reported cls/sep/mask/pad as all
        # None, and a port built against that fixture would have handled a checkpoint with no
        # special tokens (which does not exist) while missing the real ids: multilingual reuses
        # `<bos>` as CLS and `<eos>` as SEP, which nothing about the tokenizer itself reveals.
        directory = os.path.join(root, "tokenizer")
        if not os.path.isfile(os.path.join(directory, "tokenizer.json")):
            found = [os.path.join(d, "tokenizer.json")
                     for d, _subdirs, files in os.walk(root) if "tokenizer.json" in files]
            if not found:
                out[name] = {"skipped": "no tokenizer.json under %s" % root}
                continue
            directory = os.path.dirname(sorted(found, key=len)[0])
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(directory)
        raw = json.load(open(os.path.join(directory, "tokenizer.json"), encoding="utf-8"))
        out[name] = {
            "identity": {
                "tokenizer_dir": os.path.relpath(directory, rig),
                "model_type": raw["model"]["type"],
                "vocab_size": len(raw["model"].get("vocab", {})),
                "merges": len(raw["model"].get("merges", [])),
                "pre_tokenizer": (raw.get("pre_tokenizer") or {}).get("type"),
                "normalizer": (raw.get("normalizer") or {}).get("type"),
                "unk_token": raw["model"].get("unk_token"),
                "added_tokens": len(raw.get("added_tokens") or []),
                # The english checkpoint keeps 88 of its added tokens OUT of `model.vocab`, with
                # ids above the vocabulary's own maximum -- `[MASK]` among them. A port that built
                # its id map from `model.vocab` alone would have no id for masked prediction, so
                # the count is part of the checkpoint's identity rather than a detail.
                "added_tokens_outside_vocab": sum(
                    1 for a in (raw.get("added_tokens") or [])
                    if a["content"] not in raw["model"].get("vocab", {})),
                "byte_fallback": raw["model"].get("byte_fallback"),
                "ignore_merges": raw["model"].get("ignore_merges"),
                "fuse_unk": raw["model"].get("fuse_unk"),
            },
            "special_ids": {
                "cls": tok.cls_token_id, "sep": tok.sep_token_id,
                "mask": tok.mask_token_id, "pad": tok.pad_token_id,
                "mask_token": tok.mask_token,
            },
            # add_special_tokens=False everywhere: `build_sequence` adds [CLS]/[SEP] itself, so the
            # contract a port must match is the bare encoding.
            "corpus": {cid: encode_text(tok, text, add_special_tokens=False)["input_ids"]
                       for cid, text in TOKENIZER_CORPUS},
            # the 48-token cap `build_head` applies, and the same text uncapped, so a port cannot
            # pass by slicing after the fact
            "option_truncation": {
                "text": LONG_OPTION,
                "uncapped": encode_text(tok, " " + LONG_OPTION,
                                        add_special_tokens=False)["input_ids"],
                "capped_48": encode_text(tok, " " + LONG_OPTION, add_special_tokens=False,
                                         truncation=True, max_length=48)["input_ids"],
            },
        }
    # The corpus TEXT, once, beside the per-checkpoint ids. A port needs the inputs as well as the
    # expected outputs, and transcribing 37 hostile strings -- 2,000-character runs, lone
    # surrogates' worth of zero-width marks, RTL overrides -- into a second language by hand is
    # exactly how a parity suite ends up asserting against a corpus that is not the one Python
    # measured. Shared across checkpoints because both encode the same inputs.
    out["corpus_text"] = {cid: text for cid, text in TOKENIZER_CORPUS}
    return out


# Sequence cases chosen to exercise every branch of `build_head`'s budget arithmetic and every
# rendering rule, not to read nicely. `MASKS` is substituted per checkpoint so one case can test
# that each model scrubs ITS OWN mask string: the english checkpoint masks with `[MASK]` and the
# multilingual one with `<mask>`, and a port that hard-codes either lets a caller inject option
# markers into the other.
SEQUENCE_CASES = [
    ("choice-basic", "The customer was billed twice in March.",
     {"t": "choice", "ins": "What should we do?",
      "crit": {"refund": "send the money back", "replace": "ship a new unit"}}, {}),
    ("choice-no-description", "Billed twice.",
     {"t": "choice", "ins": "Approve?", "crit": {"yes": None, "no": ""}}, {}),
    ("choice-falsy-criteria", "Billed twice.",
     {"t": "choice", "ins": "Pick.", "crit": {"zero": 0, "false": False, "none": None}}, {}),
    ("choice-structured-criteria", "Billed twice.",
     {"t": "choice", "ins": "Pick.",
      "crit": {"a": {"desc": "nested", "n": 1}, "b": [1, 2, "x"], "c": 2.5}}, {}),
    ("choice-many-options", "Billed twice.",
     {"t": "choice", "ins": "Pick one of many.",
      "crit": {("opt%02d" % i): ("criterion number %d with some words" % i) for i in range(40)}}, {}),
    ("choice-long-option", "Billed twice.",
     {"t": "choice", "ins": "Pick.",
      "crit": {"short": "ok", "long": " ".join(["verylongcriterion%d" % i for i in range(120)])}}, {}),
    # Two options whose rendered text is identical for the first 48 tokens, so the tokenizer cap
    # collapses them to the SAME span. The marker count still matches the option count, so the
    # guard in `Agent._encode_state` passes and nothing downstream can tell the question lost the
    # ability to name them apart (#538) -- only `options_distinct` records it. The difference has
    # to sit past token 48, which means it must be in the LABEL's tail: differing labels put it at
    # token 1 and the collision never happens, which an earlier version of this case got wrong.
    ("choice-colliding-options", "Billed twice.",
     {"t": "choice", "ins": "Pick.",
      "crit": {("prefix " * 60 + "one"): "d", ("prefix " * 60 + "two"): "d"}}, {}),
    ("score-levels", "The reply was polite but slow.",
     {"t": "score", "ins": "Rate the reply.", "crit": ["terrible", "poor", "fine", "good"]}, {}),
    ("score-structured", "The reply was polite but slow.",
     {"t": "score", "ins": "Rate.", "crit": [{"d": 1}, "ok", 3]}, {}),
    ("noul-default", "The invoice is dated March 2nd.",
     {"t": "noul", "ins": "The invoice is from March."}, {}),
    ("noul-criteria", "The invoice is dated March 2nd.",
     {"t": "noul", "ins": "Holds?", "crit": {"false": "no it does not", "true": "yes it does"}}, {}),
    ("noul-custom-labels", "The invoice is dated March 2nd.",
     {"t": "noul", "ins": "Holds?", "labels": {"false": "nope", "true": "yep"}}, {}),
    ("mask-in-instruction", "Billed twice.",
     {"t": "choice", "ins": "Pick MASKS one MASKS now", "crit": {"a": "x", "b": "y"}}, {}),
    ("mask-in-option", "Billed twice.",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x MASKS y", "b": "MASKS"}}, {}),
    ("mask-in-state", "state with MASKS inside it",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("long-instruction", "Billed twice.",
     {"t": "choice", "ins": "instruction words " * 300, "crit": {"a": "x", "b": "y"}}, {}),
    ("long-state", "evidence sentence number one. " * 400,
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("long-state-truncate-left", "evidence sentence number one. " * 400,
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {"truncate_left": True}),
    ("state-dict", {"order": 1, "items": ["a", "b"], "n": 2.5, "ok": True, "missing": None},
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("state-list", [1, "a", {"b": 2}, None, True, 1.5],
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    # `serialize_state` and `render_criterion` both go through `json.dumps`, so a port needs
    # Python's NUMBER formatting, not its own language's. Python writes floats with `repr`
    # (shortest round-trip, scientific outside [1e-4, 1e16) with a two-digit exponent) and
    # integers at arbitrary precision; Java's `Double.toString` disagrees on both the threshold
    # and the spelling, and JDK 17's is not even always shortest. These values make the
    # disagreement a token-id difference instead of a latent one.
    ("state-hostile-numbers",
     {"big": 1e16, "small": 1e-05, "edge": 0.0001, "third": 1.0 / 3.0, "huge": 1e23,
      "whole": 2.0, "negzero": -0.0, "max": 1e308, "denormal": 5e-324,
      "bigint": 123456789012345678901234567890, "negint": -7},
     {"t": "choice", "ins": "Pick.", "crit": {"a": 1e16, "b": 1.0 / 3.0}}, {}),
    ("state-empty", "",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("state-unicode", "\u6211\u4eec\u88ab\u91cd\u590d\u6263\u8d39 \U0001f600 \u0644\u0642\u062f \u062a\u0645",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "\u9000\u6b3e", "b": "\u0627\u0633\u062a\u0631\u062f\u0627\u062f"}}, {}),
    ("option-order-permuted", "Billed twice.",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y", "c": "z"}},
     {"option_order": [2, 0, 1]}),
    ("tiny-budget", "some state here that will not fit at all",
     {"t": "choice", "ins": "a fairly long instruction that cannot fit either",
      "crit": {"a": "alpha", "b": "beta", "c": "gamma"}},
     {"max_len": 24, "head_max_len": 16}),
    ("head-larger-than-max", "state",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}},
     {"max_len": 12, "head_max_len": 192}),
]


def sequences():
    """`build_sequence` output for every rendering and budget branch, per checkpoint.

    Ids AND markers AND stats: a marker that is off by one still produces a plausible answer, and
    the option-collision counter (`options_distinct`) is invisible from the ids alone.
    """
    import copy
    import os

    from laya.common import build_sequence

    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         "/Users/nombauser/Documents/nomba-dev/laya-e2e/checkpoints")
    out = {}
    for name in ("english", "multilingual"):
        root = os.path.join(rig, name)
        directory = os.path.join(root, "tokenizer")
        if not os.path.isfile(os.path.join(directory, "tokenizer.json")):
            out[name] = {"skipped": "no tokenizer/ under %s" % root}
            continue
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(directory)
        cfg = json.load(open(os.path.join(root, "rl_agent_config.json"), encoding="utf-8"))
        default_max = cfg.get("max_len", 512)
        default_head = cfg.get("head_max_len", 192)
        cases = {}
        for cid, state, question, kwargs in SEQUENCE_CASES:
            q = copy.deepcopy(question)
            st = state
            # one case per checkpoint substitutes that checkpoint's own mask string
            if isinstance(st, str):
                st = st.replace("MASKS", tok.mask_token)
            q["ins"] = q["ins"].replace("MASKS", tok.mask_token)
            if isinstance(q.get("crit"), dict):
                q["crit"] = {k: (v.replace("MASKS", tok.mask_token) if isinstance(v, str) else v)
                             for k, v in q["crit"].items()}
            max_len = kwargs.get("max_len", default_max)
            head_max_len = kwargs.get("head_max_len", default_head)
            ids, markers, stats, trunc = build_sequence(
                tok, st, q, max_len=max_len, head_max_len=head_max_len,
                option_order=kwargs.get("option_order"),
                truncate_left=kwargs.get("truncate_left", False),
                return_stats=True, return_truncation_stats=True)
            cases[cid] = {
                "state": st, "question": q, "max_len": max_len, "head_max_len": head_max_len,
                "option_order": kwargs.get("option_order"),
                "truncate_left": kwargs.get("truncate_left", False),
                "ids": ids, "markers": markers, "stats": stats, "truncation": trunc,
            }
        out[name] = {
            "config": {"max_len": default_max, "head_max_len": default_head,
                       "temperature": cfg.get("temperature"),
                       "temperature_by_options": cfg.get("temperature_by_options"),
                       "encoder": cfg.get("encoder")},
            "special_ids": {"cls": tok.cls_token_id, "sep": tok.sep_token_id,
                            "mask": tok.mask_token_id, "pad": tok.pad_token_id,
                            "mask_token": tok.mask_token},
            "cases": cases,
        }
    return out


def decode_answers():
    """`Agent._decode_answers` output for hostile logit rows, per checkpoint.

    Driven through the REAL method rather than a transcription of it: the expectations have to come
    from the shipped decoder, including its rounding, its argmax tie-breaking and its clamps.

    Logits are synthesised rather than taken from a forward pass so the case list can aim at the
    branches -- every temperature bucket, a tie at the top, a row whose softmax is exactly uniform,
    magnitudes that would overflow a naive `exp`, and the language override -- and so the fixture
    does not need a 1.2 GB graph to regenerate.
    """
    import os

    import numpy as np

    from laya.agent import Agent, QTYPES
    from laya.common import clamp_temperature, resolve_lang_temperatures

    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         "/Users/nombauser/Documents/nomba-dev/laya-e2e/checkpoints")

    def row(values):
        return np.asarray(values, dtype=np.float32)

    # (id, question, option_order, lang, logits)
    cases = [
        ("choice-2", {"t": "choice", "ins": "i", "crit": {"a": "x", "b": "y"}},
         None, None, [2.0, 1.0]),
        ("choice-2-tied", {"t": "choice", "ins": "i", "crit": {"a": "x", "b": "y"}},
         None, None, [1.0, 1.0]),
        ("choice-4", {"t": "choice", "ins": "i",
                      "crit": {"a": "1", "b": "2", "c": "3", "d": "4"}},
         None, None, [0.5, 2.5, -1.0, 2.5]),
        ("choice-4-permuted", {"t": "choice", "ins": "i",
                               "crit": {"a": "1", "b": "2", "c": "3", "d": "4"}},
         [2, 0, 3, 1], None, [0.5, 2.5, -1.0, 2.5]),
        ("choice-7", {"t": "choice", "ins": "i",
                      "crit": {("o%d" % i): str(i) for i in range(7)}},
         None, None, [0.1 * i for i in range(7)]),
        ("choice-12", {"t": "choice", "ins": "i",
                       "crit": {("o%d" % i): str(i) for i in range(12)}},
         None, None, [(-1.0) ** i * i * 0.3 for i in range(12)]),
        ("choice-20-uniform", {"t": "choice", "ins": "i",
                               "crit": {("o%d" % i): str(i) for i in range(20)}},
         None, None, [0.0] * 20),
        ("choice-huge-magnitude", {"t": "choice", "ins": "i",
                                   "crit": {"a": "1", "b": "2", "c": "3"}},
         None, None, [700.0, 699.0, -700.0]),
        ("choice-tiny-spread", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, None, [1.0, 1.0 + 1e-7]),
        ("choice-lang-override", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, "zh-Hans", [2.0, 1.0]),
        ("choice-lang-override-prefix", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, "ZH", [2.0, 1.0]),
        ("choice-lang-unknown", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, "xx", [2.0, 1.0]),
        ("score-3", {"t": "score", "ins": "i", "crit": ["bad", "ok", "good"]},
         None, None, [0.2, 1.4, 0.9]),
        ("score-6", {"t": "score", "ins": "i", "crit": [str(i) for i in range(6)]},
         None, None, [0.0, 1.0, 2.0, 1.0, 0.0, -1.0]),
        ("score-structured-legend", {"t": "score", "ins": "i", "crit": [{"d": 1}, "ok", 3]},
         None, None, [0.4, 0.4, 0.4]),
        ("score-permuted", {"t": "score", "ins": "i", "crit": ["bad", "ok", "good"]},
         [1, 2, 0], None, [0.2, 1.4, 0.9]),
        ("noul-true", {"t": "noul", "ins": "i"}, None, None, [0.3, 1.9]),
        ("noul-false", {"t": "noul", "ins": "i"}, None, None, [2.4, 0.1]),
        ("noul-balanced", {"t": "noul", "ins": "i"}, None, None, [1.0, 1.0]),
        ("noul-labels", {"t": "noul", "ins": "i",
                         "labels": {"false": "nope", "true": "yep"}}, None, None, [0.3, 1.9]),
    ]

    out = {}
    for name in ("english", "multilingual"):
        path = os.path.join(rig, name, "rl_agent_config.json")
        if not os.path.isfile(path):
            out[name] = {"skipped": "no rl_agent_config.json for %s" % name}
            continue
        cfg = json.load(open(path, encoding="utf-8"))
        # Clamped exactly as the runtime clamps: `choice:11+` ships at 0.1006 on one checkpoint,
        # a ~10x sharpener, and a port that applied the raw value would publish a coin flip as a
        # certainty. The fixture therefore records the CLAMPED table the decoder actually uses.
        temperature = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
        by_options = {k: clamp_temperature(v)
                      for k, v in (cfg.get("temperature_by_options") or {}).items()}
        # Neither shipped checkpoint sets `lang_temperatures`, so the override branch would never
        # be exercised by their configs. One is supplied here, through the same resolver the
        # runtime uses, so a port cannot skip the branch and still pass.
        lang_raw = {"zh": {"temperature": [2.0, 1.5, 3.0],
                           "temperature_by_options": {"choice:2": 2.5}}}
        lang_temperatures = resolve_lang_temperatures(lang_raw, temperature)

        shim = Agent.__new__(Agent)
        shim.temperature = temperature
        shim.temperature_by_options = by_options
        shim.lang_temperatures = lang_temperatures
        shim.binning_map = None

        recorded = {}
        for cid, question, option_order, lang, logits in cases:
            k = len(logits)
            q = dict(question)
            if option_order is not None:
                q["option_order"] = option_order
            items = [{"markers": list(range(k))}]
            # a wider logits block than k, so a port that forgets to slice to k columns is caught
            block_width = k + 3
            padded = row(list(logits) + [99.0] * 3).reshape(1, block_width)
            act = row([0.25, -0.5]).reshape(1, 2)
            answers = Agent._decode_answers(shim, padded, act, items, [cid], {cid: q}, 0, lang)
            recorded[cid] = {
                "question": question, "option_order": option_order, "lang": lang,
                "logits": [float(x) for x in logits], "logits_block_width": block_width,
                "act": [0.25, -0.5], "answer": answers[cid],
            }
        out[name] = {
            "config": {"temperature": temperature, "temperature_by_options": by_options,
                       "lang_temperatures_raw": lang_raw},
            "qtypes": QTYPES,
            "cases": recorded,
        }
    return out


def python_json():
    """CPython's `json.dumps(ensure_ascii=False)` and `round(v, 4)` on values a port gets wrong.

    Needs no checkpoint, so these are the parity tests CI can run without downloading a model.

    Doubles are carried as raw 64-bit patterns rather than as decimal text: the point of the
    fixture is the SPELLING of a double, so round-tripping the operand through a decimal literal
    would lose the very thing being measured.

    **Nothing here may go through libm.** `math.exp`, and `10 ** n` for a large negative n, are C
    library calls, and those are not bit-identical across platforms: this family regenerated
    differently on Linux than on macOS and CI reported the committed copy as stale. Every value is
    now produced by an exact route -- a decimal literal (correctly rounded by strtod), IEEE
    arithmetic, integer division, or a raw bit pattern -- so the file is a property of the seed
    rather than of the machine that ran the generator.
    """
    import random
    import struct

    def bits(value):
        return struct.unpack("<Q", struct.pack("<d", value))[0]

    # Values where Java's own spelling differs from Python's, plus the boundaries of the
    # fixed/scientific switch, the subnormal floor, and both infinities.
    hostile = [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 1.0 / 3, 2.0 / 3, 0.1, 0.2, 0.3,
               1e-5, 1e-4, 0.0001, 1e15, 1e16, 1e17, 1e23, 1e100, 1e-100, 1e308, 5e-324,
               2.2250738585072014e-308, 1.7976931348623157e308, 9007199254740993.0,
               1e7, 1e-3, 123456789.123456789,
               # pi and e as literals rather than `math.pi` / `math.e`: the constants are
               # identical everywhere, but writing them out keeps this family free of any `math`
               # reference at all, which is the rule the docstring states.
               3.141592653589793, 2.718281828459045,
               float("inf"), float("-inf"), float("nan")]
    rng = random.Random(1091)
    doubles = list(hostile)
    # Enough random draws to cover the exponent range and the raw bit patterns, kept small enough
    # that the committed file stays reviewable: the hostile list above is what actually
    # discriminates a wrong implementation, and these are the sweep behind it.
    #
    # `random()` is exact -- it scales Mersenne Twister integers by a power of two -- and a raw bit
    # pattern is exact by construction and reaches the subnormals, the infinities and the NaNs that
    # arithmetic would not. The previous version multiplied by `10 ** rng.randint(-320, 300)`,
    # which is a libm `pow`, and that is what made this file machine-dependent.
    for _ in range(500):
        kind = rng.randrange(3)
        if kind == 0:
            doubles.append(rng.random())
        elif kind == 1:
            doubles.append(-rng.random())
        else:
            doubles.append(struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0])

    quote, back, newline, tab = chr(34), chr(92), chr(10), chr(9)
    structured = [
        {"b": 1, "a": 2, "z": [1, "x", None, True, 2.5]},
        {"nested": {"k": {"deep": [1.5, -0.0]}}},
        ["a", "b" + quote + "c", "d" + back + "e", "f" + newline + "g" + tab + "h",
         chr(0) + chr(31) + chr(1), chr(0x2028) + chr(0x2029), "/",
         chr(0xE9) + chr(0x4E2D) + chr(0x1F600)],
        {chr(0x4E2D) + chr(0x6587): chr(0x503C), "emoji " + chr(0x1F600): [1e16, 1e-05]},
        "a plain string", 123456789012345678901234567890, -7, True, None, [], {},
    ]

    # Rounding: probability-shaped values, and the exactly-representable halfway values a half-up
    # implementation gets wrong. Both groups are kept separate so a failure says which kind broke.
    #
    # The distributions are built by integer division rather than by an actual softmax: `math.exp`
    # is libm and would make the committed file machine-dependent. `weight / total` is one
    # correctly-rounded IEEE division of two exact integers, so it is identical everywhere, and it
    # still produces the normalised, unevenly-spread values a softmax produces -- which is all the
    # rounding rule cares about.
    rounding = {"distribution": [], "halfway": [], "uniform": []}
    for _ in range(100):
        k = rng.randint(2, 20)
        weights = [rng.getrandbits(20) + 1 for _ in range(k)]
        total = sum(weights)
        rounding["distribution"].extend(weight / total for weight in weights)
    # Every 53rd step rather than every 7th: the step size does not matter, only that the values
    # are exact halves at the fourth decimal, and roughly a fifth of them discriminate half-up from
    # half-even -- which is ample at this size.
    rounding["halfway"] = [i / 20000.0 for i in range(0, 20000, 53)]
    rounding["uniform"] = [rng.random() for _ in range(300)]

    return {
        "doubles": [{"bits": bits(v), "repr": json.dumps(v)} for v in doubles],
        "structured": [{"value": v, "dumps": json.dumps(v, ensure_ascii=False)}
                       for v in structured],
        "round4": {name: [{"bits": bits(v), "r4": round(v, 4)} for v in values]
                   for name, values in rounding.items()},
    }


# States and questions for the model-backed golden. Public question schema, because this one goes
# through `ONNXAgent.predict` rather than straight into `build_sequence`.
PREDICT_CASES = [
    ("billing-mixed", "We were billed twice for March and want a refund today.", None, {
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"refund": "money back for a duplicate charge",
                                "replace": "a replacement unit", "info": "an explanation only"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "somewhat", "urgent", "critical"]},
        "duplicate": {"type": "noul", "instructions": "The customer was charged more than once."},
    }),
    ("cjk", "\u6211\u4eec\u4e09\u6708\u4efd\u88ab\u91cd\u590d\u6263\u8d39\u4e86\u4e24\u6b21\uff0c\u8bf7\u4eca\u5929\u9000\u8fd8\u3002", "zh", {
        "intent": {"type": "choice", "instructions": "\u5ba2\u6237\u60f3\u8981\u4ec0\u4e48\uff1f",
                   "criteria": {"refund": "\u9000\u6b3e", "replace": "\u66f4\u6362"}},
        "holds": {"type": "noul", "instructions": "\u5ba2\u6237\u88ab\u91cd\u590d\u6263\u8d39\u3002"},
    }),
    ("many-options", "The reply was polite but arrived nine days late.", None, {
        "grade": {"type": "choice", "instructions": "Grade the reply.",
                  "criteria": {("g%02d" % i): ("grade band number %d" % i) for i in range(14)}},
    }),
    ("structured-state", {"order": 1182, "items": ["widget", "case"], "charged": 2,
                          "currency": "NGN", "note": None}, None, {
        "double": {"type": "noul", "instructions": "This order was charged twice."},
        "band": {"type": "score", "instructions": "Rate the severity.",
                 "criteria": [{"d": "none"}, "minor", 3]},
    }),
    ("long-state", "evidence sentence number one about the duplicate charge. " * 120, None, {
        "intent": {"type": "choice", "instructions": "What should we do?",
                   "criteria": {"refund": "send money back", "escalate": "pass to a human"}},
    }),
    ("unicode-mixed", "Refund \u9000\u6b3e \u0627\u0633\u062a\u0631\u062f\u0627\u062f \u0935\u093e\u092a\u0938\u0940 \U0001f600 now!", "ar", {
        "intent": {"type": "choice", "instructions": "Pick one.",
                   "criteria": {"a": "first", "b": "second"}},
    }),
]


def predict_golden():
    """`ONNXAgent.predict` and `predict_batch` output on a real graph, for the end-to-end gate.

    Needs an exported graph, which this repository does not ship, so it records `skipped` when
    `LAYA_ONNX_GRAPH` is unset and the Java test skips in turn. CI exports the checkpoint and
    regenerates this, the way the .NET lane regenerates its goldens, so a Python-side change shows
    up here as drift rather than as silence.

    The numbers are model-derived and therefore platform-sensitive in their last reported digit.
    The Java test compares structure exactly and probabilities within a stated tolerance; it is the
    test's tolerance that is the gate, not byte equality of this file.
    """
    import os

    graph = os.environ.get("LAYA_ONNX_GRAPH")
    model = os.environ.get("LAYA_PREDICT_MODEL", "multilingual")
    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         "/Users/nombauser/Documents/nomba-dev/laya-e2e/checkpoints")
    if not graph or not os.path.exists(graph):
        return {"skipped": "set LAYA_ONNX_GRAPH to an exported laya.onnx to record this family"}
    from laya.onnx_agent import ONNXAgent
    agent = ONNXAgent(os.path.join(rig, model), onnx_path=graph)
    single = {}
    for cid, state, lang, questions in PREDICT_CASES:
        result = agent.predict(state, questions, lang=lang)
        single[cid] = {"state": state, "lang": lang, "questions": questions,
                       "model": result["model"], "answers": result["answers"],
                       "usage": result["usage"]}
    batch_states = [c[1] for c in PREDICT_CASES] + ["short one", "another short state", ""]
    batch_questions = {
        "intent": {"type": "choice", "instructions": "What should we do?",
                   "criteria": {"refund": "send money back", "escalate": "pass to a human",
                                "ignore": "no action"}},
        "urgent": {"type": "noul", "instructions": "This needs a human today."},
    }
    batches = []
    for batch_size, sort_by_length in ((None, False), (2, False), (3, True)):
        kwargs = {"sort_by_length": sort_by_length}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        results = agent.predict_batch(batch_states, batch_questions, **kwargs)
        batches.append({"batch_size": batch_size, "sort_by_length": sort_by_length,
                        "states": batch_states, "questions": batch_questions,
                        "results": [{"model": r["model"], "answers": r["answers"],
                                     "usage": r["usage"]} for r in results]})
    return {"checkpoint": model, "graph": os.path.basename(graph),
            "single": single, "batch": batches}


FAMILIES = {
    "lang_tables.json": lang_tables,
    "presets.json": presets,
    "tokenizer_ids.json": tokenizer_ids,
    "sequences.json": sequences,
    "decode.json": decode_answers,
    "python_json.json": python_json,
    "predict.json": predict_golden,
}


def render(payload):
    """One canonical serialisation, so `--check` compares content and never formatting.

    `sort_keys` is deliberately OFF. Key order is not formatting here, it is data: a `choice`
    question's options are rendered in the criteria map's INSERTION order, so sorting the keys on
    the way into the fixture asks a different question than the one that was measured, and a state
    dict re-serialised in sorted order tokenizes differently. Sorting them made the committed
    fixture disagree with the library it was generated from, and a port that matched the fixture
    was wrong in exactly the way the fixture existed to prevent. Python dicts preserve insertion
    order and this generator builds them deterministically, so the output is still canonical.
    """
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def unverifiable(payload):
    """True when regeneration produced nothing but skip markers.

    A family that needs a checkpoint or an exported graph records `{"skipped": ...}` when it cannot
    reach one. That must never be mistaken for data:

    * in write mode it would REPLACE a good committed fixture with a skip marker, so running this
      script on a machine without the checkpoints would silently delete the parity expectations;
    * in `--check` mode it would be reported as drift against the committed file, so the gate would
      fail for the one reason that is not a problem.

    Both are refused below. `--strict` turns "could not verify" into an error, which is what a CI
    lane that DID download the checkpoints should pass -- there, a skip means the download failed.
    """
    if not isinstance(payload, dict):
        return False
    if "skipped" in payload:
        return True
    # ANY section, not all of them. `tokenizer_ids.json` carries a `corpus_text` section that is
    # always present, so requiring every section to be skipped let a payload through whose two
    # per-checkpoint halves were both skip markers -- and the write replaced a 107 KB fixture with
    # a 9 KB one. Partial regeneration is still unsafe to write.
    return any("skipped" in value
               for value in payload.values() if isinstance(value, dict))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if a committed fixture would change")
    parser.add_argument("--only", action="append", metavar="FILE",
                        help="restrict to one fixture file (repeatable)")
    parser.add_argument("--strict", action="store_true",
                        help="treat a family that cannot be regenerated as an error, for a lane "
                             "that is supposed to have the checkpoints")
    args = parser.parse_args(argv)

    os.makedirs(FIXTURES, exist_ok=True)
    stale, written, unverified = [], [], []
    for filename, build in sorted(FAMILIES.items()):
        if args.only and filename not in args.only:
            continue
        path = os.path.join(FIXTURES, filename)
        payload = build()
        fresh = render(payload)
        if unverifiable(payload) and os.path.exists(path):
            unverified.append(filename)
            continue
        if args.check:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    current = handle.read()
            except OSError:
                stale.append("%s is missing" % filename)
                continue
            if current != fresh:
                stale.append("%s is stale" % filename)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(fresh)
            written.append("%s (%d bytes)" % (filename, len(fresh)))

    for filename in unverified:
        print("gen_fixtures: %s left alone: it needs a checkpoint or an exported graph this run "
              "could not reach" % filename, file=sys.stderr)
    if unverified and args.strict:
        print("gen_fixtures: --strict was asked for, so a family this run could not regenerate is "
              "an error: the checkpoints were expected to be present", file=sys.stderr)
        return 1

    if args.check:
        for line in stale:
            print("gen_fixtures: " + line, file=sys.stderr)
        if stale:
            print("gen_fixtures: run laya-java/scripts/gen_fixtures.py and commit the result",
                  file=sys.stderr)
            return 1
        print("gen_fixtures: fixtures match laya%s"
              % (" (%d family/families unverified)" % len(unverified) if unverified else ""))
        return 0
    for line in written:
        print("wrote " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
