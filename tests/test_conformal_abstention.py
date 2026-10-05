"""Per-bucket abstention thresholds (#394): fit one cut per option-count bucket so a single
confidence threshold's failure to transfer across option counts is fixed.

Weight-free: synthetic (qtype, logits, target, k) records with deliberately different confidence
regimes per bucket, and synthetic answer dicts for the gate. No model, no download.

Run: python tests/test_conformal_abstention.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.calibrate import (  # noqa: E402
    apply_binning_map,
    fit_abstention_thresholds,
    fit_binning_map,
)
from laya.confidence import (  # noqa: E402
    check_min_confidence,
    check_min_confidence_map,
    resolve_min_confidence,
    flag_low_confidence,
    apply_confidence_gate,
    GATE_ABSTAINED,
    GATE_PASSED,
)

PASS, FAIL = [], []


def check(name, got, want):
    (PASS if got == want else FAIL).append(name if got == want else "%s: got %r want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
    except Exception as e:  # noqa: BLE001
        FAIL.append("%s: raised %r not %s" % (name, e, exc.__name__))
    else:
        FAIL.append("%s: did not raise %s" % (name, exc.__name__))


# ---- build a (qtype, logits, target, k) record realising a given (k, confidence, correct) ----
def record(k, conf, correct, qtype=0):
    conf = min(max(conf, 1e-3), 1 - 1e-3)
    L = math.log(conf * (k - 1) / (1 - conf))   # logit on index 0 so softmax(z).max() == conf
    z = [0.0] * k
    z[0] = L
    target = [0.0] * k
    target[0 if correct else 1] = 1.0           # argmax(z)==0; correct iff gold is 0
    return (qtype, z, target, k)


# Two choice buckets with DIFFERENT confidence regimes:
#   k=2  : overconfident -- a 0.85-confidence cohort is only half right, so a low cut lets errors in.
#   k=12 : well separated -- a 0.50-confidence cohort is already right, so a low cut is safe.
records = []
for _ in range(120):
    records.append(record(2, 0.97, True))        # k=2 confident & correct
for i in range(120):
    records.append(record(2, 0.85, i % 2 == 0))  # k=2 confident but 50% wrong
for _ in range(120):
    records.append(record(12, 0.55, True))       # k=12 modest conf, correct
for i in range(120):
    records.append(record(12, 0.33, i % 5 == 0)) # k=12 low conf, 80% wrong

TARGET = 0.10
thr = fit_abstention_thresholds(records, [1.0, 1.0, 1.0], {}, target_error=TARGET, min_bucket_n=50)
check_true("fit/keyed by temp_bucket", set(thr) == {"choice:2", "choice:11+"}, thr)
check_true("fit/k=2 needs a high cut", thr["choice:2"] > 0.9, thr)
check_true("fit/k=12 accepts a lower cut", thr["choice:11+"] < 0.6, thr)
check_true("fit/the two cuts genuinely differ", thr["choice:2"] - thr["choice:11+"] > 0.3, thr)


# ---- apply per-bucket vs one global cut, measure error among accepted per bucket ----
def err_and_cov(records, cut_fn):
    out = {}
    for qt, z, t, k in records:
        import numpy as np
        p = np.exp(np.array(z) - max(z)); p = p / p.sum()
        conf = float(p.max()); correct = int(p.argmax()) == int(np.argmax(t))
        key = "choice:2" if k <= 2 else "choice:11+"
        if conf >= cut_fn(key):
            a, e = out.get(key, (0, 0))
            out[key] = (a + 1, e + (0 if correct else 1))
    return out

per_bucket = err_and_cov(records, lambda key: thr[key])
for key, (acc, err) in per_bucket.items():
    rate = err / acc if acc else 0.0
    check_true("apply/per-bucket holds target in %s (err %.0f%%, n=%d)" % (key, rate * 100, acc),
               acc > 0 and rate <= TARGET + 0.02, (acc, err))

# one global cut = the per-bucket cut for k=2 (0.9+); it over-abstains k=12 to near-zero coverage,
# which is exactly the #394 failure: a cut tuned for one option count does not serve the other.
GLOBAL = thr["choice:2"]
glob = err_and_cov(records, lambda key: GLOBAL)
k12_acc = glob.get("choice:11+", (0, 0))[0]
check_true("transfer/global k=2 cut starves k=12 coverage (#394)", k12_acc == 0, glob)
# and the mirror: the k=12 cut lets k=2 errors through
GLOBAL2 = thr["choice:11+"]
glob2 = err_and_cov(records, lambda key: GLOBAL2)
k2_acc, k2_err = glob2.get("choice:2", (0, 0))
check_true("transfer/global k=12 cut exceeds target error on k=2 (#394)",
           k2_acc > 0 and (k2_err / k2_acc) > TARGET, (k2_acc, k2_err))


# ---- confidence.py: resolve + gate with a map ----
def ans(qtype, k, conf):
    probs = {str(i): 0.0 for i in range(k)}
    return {"type": qtype, "probabilities": probs, "answer_confidence": conf}

MAP = {"choice:2": 0.9, "choice:11+": 0.5, "noul:2": 0.8, "default": 0.3}
check("resolve/choice k=2 bucket", resolve_min_confidence(ans("choice", 2, 0.5), MAP), 0.9)
check("resolve/choice k=12 bucket (11+)", resolve_min_confidence(ans("choice", 12, 0.5), MAP), 0.5)
check("resolve/choice k=4 falls to default", resolve_min_confidence(ans("choice", 4, 0.5), MAP), 0.3)
check("resolve/noul bucket", resolve_min_confidence({"type": "noul", "answer_confidence": 0.5}, MAP), 0.8)
check("resolve/unknown + no default -> 0.0 (gate nothing)",
      resolve_min_confidence(ans("score", 7, 0.5), {"choice:2": 0.9}), 0.0)

# gate a batch with a per-bucket map: a 0.85-confident k=2 answer abstains (cut 0.9), a 0.6-confident
# k=12 answer passes (cut 0.5) -- a single float could not do both.
results = [{"answers": {"a": ans("choice", 2, 0.85), "b": ans("choice", 12, 0.60)}}]
apply_confidence_gate(results, MAP)
check("gate/k=2 low-conf abstains under its bucket cut", results[0]["answers"]["a"]["abstention"], GATE_ABSTAINED)
check("gate/k=12 passes under its lower bucket cut", results[0]["answers"]["b"]["abstention"], GATE_PASSED)
check("gate/echoes the per-bucket threshold (k=2)", results[0]["answers"]["a"]["abstention_threshold"], 0.9)
check("gate/echoes the per-bucket threshold (k=12)", results[0]["answers"]["b"]["abstention_threshold"], 0.5)

# the float path is unchanged
res_f = [{"answers": {"a": ans("choice", 3, 0.4)}}]
flag_low_confidence(res_f, 0.7)
check("float/below a scalar threshold still flags", res_f[0]["answers"]["a"].get("low_confidence"), True)
check("validate/float still accepted", check_min_confidence(0.7), 0.7)
check("validate/map accepted and normalised", check_min_confidence({"choice:2": 0.9}), {"choice:2": 0.9})
check_raises("validate/map value out of range rejected", ValueError,
             lambda: check_min_confidence_map({"choice:2": 1.5}))
check_raises("validate/map non-string key rejected", ValueError,
             lambda: check_min_confidence_map({2: 0.9}))



# --------------------------------------------------------------- thresholds vs a binning map
# A histogram-binning map recalibrates `answer_confidence` at runtime (`Agent._decode_answers`),
# so a cut fitted on the temperature-scaled scale gates a quantity the runtime no longer reports:
# the same number admits far more than its target error. Fitting with the map puts both on one
# scale, and then the order the two were fitted in stops mattering.
import numpy as np  # noqa: E402

_rng = np.random.default_rng(7)
_binning_records = []
for _ in range(1200):
    _true = int(_rng.integers(12))
    _z = _rng.normal(0.0, 1.0, 12)
    _z[_true] += 2.2                        # usually right, sometimes not: a continuous spread
    _t = [0.0] * 12
    _t[_true] = 1.0
    _binning_records.append((0, _z.tolist(), _t, 12))

_temps, _tbo = [1.0, 1.0, 1.0], {}
_bmap = fit_binning_map(_binning_records, _temps, _tbo)
_cut_raw = fit_abstention_thresholds(_binning_records, _temps, _tbo, target_error=0.10)["choice:11+"]
_cut_binned = fit_abstention_thresholds(_binning_records, _temps, _tbo, binning_map=_bmap,
                                        target_error=0.10)["choice:11+"]


def _gate_error(cut, binned):
    """Coverage and error among accepted, when the runtime reports binned (or raw) confidences."""
    accepted = wrong = 0
    for qt, z, t, k in _binning_records:
        p = np.exp(np.asarray(z) - max(z))
        p = p / p.sum()
        conf = float(p.max())
        if binned:
            conf = apply_binning_map(conf, "choice:11+", _bmap)
        if conf >= cut:
            accepted += 1
            wrong += int(int(p.argmax()) != int(np.argmax(t)))
    return accepted, (wrong / accepted if accepted else 0.0)


_acc_raw, _err_raw = _gate_error(_cut_raw, binned=False)
_acc_mismatch, _err_mismatch = _gate_error(_cut_raw, binned=True)
_acc_fit, _err_fit = _gate_error(_cut_binned, binned=True)

check_true("binning/an un-binned cut holds its target on its own scale", _err_raw <= 0.12,
           "cut=%.4f accepted=%d error=%.3f" % (_cut_raw, _acc_raw, _err_raw))
check_true("binning/the same cut stops gating once the runtime bins", _err_mismatch > 0.2,
           "cut=%.4f accepted=%d error=%.3f" % (_cut_raw, _acc_mismatch, _err_mismatch))
check_true("binning/fitting with the map restores the target", _err_fit <= 0.12,
           "cut=%.4f accepted=%d error=%.3f" % (_cut_binned, _acc_fit, _err_fit))
check_true("binning/the two cuts genuinely differ", abs(_cut_binned - _cut_raw) > 1e-9,
           "raw=%.4f binned=%.4f" % (_cut_raw, _cut_binned))
check("binning/no map leaves the fit unchanged",
      fit_abstention_thresholds(_binning_records, _temps, _tbo, binning_map=None,
                                target_error=0.10)["choice:11+"], _cut_raw)
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
