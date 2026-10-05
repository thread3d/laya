"""The secret scan's allowlist must stay both narrow and readable by the pinned gitleaks.

The weekly `Security` run scans all history, so it keeps re-finding the two `generic-api-key`
false positives the zh_reliability benchmark pinned:

    "tokenizer/tokenizer.json": "<64 lowercase hex>"
    "tokenizer/tokenizer_config.json": "<64 lowercase hex>"

Those values are the SHA-256 digests of the model files the benchmark exists to record, so the
revert that pulled the benchmark out of the release (2c73632) removed the file from HEAD but not
from history, and `.gitleaks.toml` has to excuse them.

Two properties of that file are easy to break and hard to notice, because the failure only shows
up on the weekly schedule:

* The action pins gitleaks 8.24.3, which reads the singular `[allowlist]` table and *silently
  ignores* the newer `[[allowlists]]` array. "Modernizing" the config therefore turns the scan
  red again with no local signal.
* On 8.24.3 an entry carrying both `paths` and `regexes` is an OR, not an AND. A path entry would
  excuse any secret ever committed under that path, so the entry names no paths and instead
  matches the one benign shape.

This suite pins both properties and the regex's precision: it matches the pinned digest lines and
nothing that merely looks like a key.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s: %s" % (name, detail))


with open(os.path.join(ROOT, ".gitleaks.toml"), encoding="utf-8") as fh:
    config = fh.read()

# The comments above name the very spellings this suite rejects, so the structural checks read
# the code lines only -- otherwise the explanation of the trap would trip the trap's own guard.
code = "\n".join(l for l in config.splitlines() if not l.lstrip().startswith("#"))

# The default ruleset has to survive: the allowlist is a record of reviewed false positives, not
# a replacement for the rules that catch everything else.
check_true("gitleaks/config extends the default ruleset",
           re.search(r"(?m)^\[extend\]\s*$", code) is not None
           and re.search(r"(?m)^\s*useDefault\s*=\s*true\s*$", code) is not None,
           "without [extend] useDefault = true the file disables every default rule")

# The action pins 8.24.3, which predates `[[allowlists]]` and drops it on the floor.
check_true("gitleaks/uses the singular [allowlist] table the pinned action reads",
           re.search(r"(?m)^\[allowlist\]\s*$", code) is not None
           and "[[allowlists]]" not in code,
           "gitleaks 8.24.3 silently ignores [[allowlists]]; only [allowlist] takes effect")

# A path entry degrades to an OR on 8.24.3, which would hide any secret in that file.
check_true("gitleaks/scopes by the line shape, never by path",
           re.search(r"(?m)^\s*paths\s*=", code) is None
           and re.search(r'(?m)^\s*regexTarget\s*=\s*"line"\s*$', code) is not None,
           "8.24.3 treats paths+regexes as OR; the digest shape is the only benign thing")

# The digests the benchmark pinned, assembled from fragments: this file is scanned like any
# other, and a raw 64-hex literal under a tokenizer key would be a finding in its own right.
DIGEST = "609d8f4c" + "067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"
DIGEST_CONFIG = "424b6944" + "4bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7"


def json_pair(key, value):
    return '"%s": "%s"' % (key, value)


regexes = re.findall(r"'''(.*?)'''", code, re.S)
check_true("gitleaks/names exactly one allowlist pattern", len(regexes) == 1, regexes)

if len(regexes) == 1:
    pattern = re.compile(regexes[0])

    # The two findings the weekly run reports, as they appear in the artifact.
    allowed = [
        json_pair("tokenizer/tokenizer.json", DIGEST),
        json_pair("tokenizer/tokenizer_config.json", DIGEST_CONFIG),
    ]
    for line in allowed:
        check_true("gitleaks/allows the pinned digest %s" % line[:40],
                   pattern.search(line) is not None, "the weekly run will fail again")

    # Near misses that must stay findings. If any of these matches, the allowlist has grown past
    # the false positive it exists for. Key and value are assembled here rather than written as
    # one literal, so the scanner this file configures does not flag the fakes themselves.
    rejected = {
        "a non-hash tokenizer value": json_pair("tokenizer/tokenizer.json", "hunter" + "2"),
        "a 64-hex value under a different key": json_pair("api" + "_key", DIGEST),
        "uppercase hex": json_pair("tokenizer/tokenizer.json", DIGEST.upper()),
        "a 63-character value": json_pair("tokenizer/tokenizer.json", DIGEST[:-1]),
        "a tokenizer key that is not tokenizer.json": json_pair("tokenizer/merges.txt", DIGEST),
    }
    for label, line in rejected.items():
        check_true("gitleaks/rejects %s" % label,
                   pattern.search(line) is None, "the allowlist would hide a real finding")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for failure in FAIL:
    print("  FAIL " + failure)
sys.exit(1 if FAIL else 0)
