"""laya.train: data prep, option-order augmentation, losses and an end-to-end run.

CPU only and weight-free: the checkpoint is a one-layer BERT with a word-level tokenizer built
in a temp directory, so nothing is downloaded.

Run: python tests/test_train.py
"""
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya import load  # noqa: E402
from laya.common import (  # noqa: E402
    DecisionModel,
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    build_sequence,
    proper_reward,
    render_options,
    unpermute_probs,
)
from laya.train import (  # noqa: E402
    TrainConfig,
    draw_option_order,
    encode_item,
    encode_state,
    finetune,
    items_from_rows,
    make_item,
    read_data,
    rows_from_csv,
    rlcd_loss,
    soft_ce_loss,
    split_calibration,
    target_from_expected,
    target_from_gold,
    to_internal,
    train_model,
)
from laya.train_cli import build_parser as build_train_parser, main as train_cli_main  # noqa: E402

WORDS = ["refund", "invoice", "charged", "twice", "crash", "error", "app", "screen", "please",
         "billing", "technical", "other", "is", "it", "urgent", "low", "high", "level", "question",
         "choice", "score", "noul", "which", "team", "yes", "no", "the", "statement", "does",
         "not", "hold", "holds", "one", "two", "three", "four", "five", "six", "seven", ":", ",", "?"]

DEPARTMENT = {"type": "choice", "instructions": "which team ?",
              "criteria": {"billing": "", "technical": "", "other": ""}}
URGENT = {"type": "noul", "instructions": "is it urgent ?"}
LEVEL = {"type": "score", "instructions": "which level ?", "criteria": ["low", "high"]}

BILLING_STATES = ["please refund invoice", "charged twice invoice", "refund charged twice", "invoice refund please"]
TECH_STATES = ["app crash error", "screen error crash", "crash app screen", "error screen app"]


def make_tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for w in WORDS:
        vocab[w] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def make_checkpoint(path: Path, tok):
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
    config.save_pretrained(path / "encoder")
    tok.save_pretrained(path / "tokenizer")
    model = DecisionModel(BertModel(config), head_layers=1, n_act=2)
    save_file(model.state_dict(), path / "model.safetensors")
    cfg = {"encoder": "unused/offline", "head_layers": 1, "act_costs": {"act": 0},
           "max_len": 64, "head_max_len": 40, "temperature": [1.0, 1.0, 1.0],
           "temperature_by_options": {"choice:3-5": 0.7}}
    (path / "rl_agent_config.json").write_text(json.dumps(cfg))


def rows(n_repeats=6):
    out = []
    for _ in range(n_repeats):
        for s in BILLING_STATES:
            out.append({"state": s, "questions": {"department": DEPARTMENT, "urgent": URGENT, "level": LEVEL},
                        "gold": {"department": {"probabilities": {"billing": 0.9, "technical": 0.05, "other": 0.05}},
                                 "urgent": {"probabilities": {"false": 0.8, "true": 0.2}},
                                 "level": {"probabilities": {"0": 0.7, "1": 0.3}}}})
        for s in TECH_STATES:
            out.append({"state": s, "questions": {"department": DEPARTMENT, "urgent": URGENT, "level": LEVEL},
                        "gold": {"department": {"probabilities": {"billing": 0.05, "technical": 0.9, "other": 0.05}},
                                 "urgent": {"probabilities": {"false": 0.2, "true": 0.8}},
                                 "level": {"probabilities": {"0": 0.2, "1": 0.8}}}})
    return out


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = make_tokenizer()

    def item(self, question, target, state="app crash", head_max_len=40):
        item, reason = make_item(self.tok, to_internal("q", question), target,
                                 encode_state(self.tok, state, 64), head_max_len)
        self.assertIsNone(reason)
        return item

    def test_target_from_gold_orders_and_normalises(self):
        def target(question, probs):
            return target_from_gold(to_internal("q", question), {"probabilities": probs})

        self.assertEqual(target(DEPARTMENT, {"technical": 2.0, "billing": 2.0}), [0.5, 0.5, 0.0])
        self.assertEqual(target(URGENT, {"true": 0.75, "false": 0.25}), [0.25, 0.75])
        self.assertEqual(target(LEVEL, {"1": 1.0}), [0.0, 1.0])
        self.assertEqual(target(DEPARTMENT, {}), [1 / 3] * 3)
        # list-form choice and non-string labels, which inference accepts
        self.assertEqual(target({"type": "choice", "instructions": "which ?", "criteria": ["app", "screen"]},
                                {"screen": 1.0}), [0.0, 1.0])
        self.assertEqual(target({"type": "choice", "instructions": "which ?", "criteria": {1: "", 2: ""}},
                                {"2": 3.0, "1": 1.0}), [0.25, 0.75])
        for bad in ({"billing": -1.0}, {"billing": float("nan")}, {"billing": "high"}):
            with self.assertRaises(ValueError):
                target(DEPARTMENT, bad)

    def test_target_from_expected_choice(self):
        def target(question, exp, smoothing=0.0):
            return target_from_expected(to_internal("q", question), exp, label_smoothing=smoothing)

        self.assertEqual(target(DEPARTMENT, "billing"), [1.0, 0.0, 0.0])
        self.assertEqual(target(DEPARTMENT, "technical"), [0.0, 1.0, 0.0])
        self.assertEqual(target(DEPARTMENT, "other"), [0.0, 0.0, 1.0])
        # Non-string labels matching keys
        self.assertEqual(target({"type": "choice", "instructions": "which ?", "criteria": {1: "a", 2: "b"}}, 1),
                         [1.0, 0.0])

    def test_target_from_expected_noul(self):
        def target(question, exp, smoothing=0.0):
            return target_from_expected(to_internal("q", question), exp, label_smoothing=smoothing)

        self.assertEqual(target(URGENT, True), [0.0, 1.0])
        self.assertEqual(target(URGENT, False), [1.0, 0.0])
        self.assertEqual(target(URGENT, "true"), [0.0, 1.0])
        self.assertEqual(target(URGENT, "false"), [1.0, 0.0])

    def test_target_from_expected_score(self):
        def target(question, exp, smoothing=0.0):
            return target_from_expected(to_internal("q", question), exp, label_smoothing=smoothing)

        self.assertEqual(target(LEVEL, 0), [1.0, 0.0])
        self.assertEqual(target(LEVEL, 1), [0.0, 1.0])
        self.assertEqual(target(LEVEL, "1"), [0.0, 1.0])

        # Fractional score answers (e.g. from laya-evals ScoreMAE / ScoreWithin)
        three_level = {"type": "score", "instructions": "which level ?", "criteria": ["low", "mid", "high"]}
        self.assertEqual(target(three_level, 1.5), [0.0, 0.5, 0.5])
        res_frac2 = target(three_level, 0.2)
        self.assertAlmostEqual(res_frac2[0], 0.8)
        self.assertAlmostEqual(res_frac2[1], 0.2)
        self.assertAlmostEqual(res_frac2[2], 0.0)

    def test_target_from_expected_label_smoothing(self):
        res = target_from_expected(to_internal("q", DEPARTMENT), "billing", label_smoothing=0.1)
        self.assertAlmostEqual(res[0], 0.9 + 0.1 / 3)
        self.assertAlmostEqual(res[1], 0.1 / 3)
        self.assertAlmostEqual(res[2], 0.1 / 3)
        self.assertAlmostEqual(sum(res), 1.0)

        res_noul = target_from_expected(to_internal("q", URGENT), True, label_smoothing=0.2)
        self.assertAlmostEqual(res_noul[0], 0.1)
        self.assertAlmostEqual(res_noul[1], 0.9)
        self.assertAlmostEqual(sum(res_noul), 1.0)

    def test_target_from_expected_rejects_invalid(self):
        with self.assertRaises(ValueError):
            target_from_expected(to_internal("q", DEPARTMENT), "unknown_choice")
        with self.assertRaises(ValueError):
            target_from_expected(to_internal("q", URGENT), "not_a_bool")
        for bad_num in (-3, 7, 0.5, -0.1, 2):
            with self.assertRaises(ValueError):
                target_from_expected(to_internal("q", URGENT), bad_num)
        with self.assertRaises(ValueError):
            target_from_expected(to_internal("q", LEVEL), 99)
        with self.assertRaises(ValueError):
            target_from_expected(to_internal("q", LEVEL), -0.5)
        for bad_smooth in (-0.1, 1.0, 1.5, True):
            with self.assertRaises(ValueError):
                target_from_expected(to_internal("q", DEPARTMENT), "billing", label_smoothing=bad_smooth)

    def test_resolve_checkpoint_dir(self):
        from laya.train import resolve_checkpoint_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "rl_agent_config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_checkpoint_dir(str(p)), str(p))
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint_dir(str(p / "nonexistent"))

    def test_rows_from_csv_and_read_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "tickets.csv"
            csv_path.write_text("body,department\nrefund my invoice,billing\napp crash on startup,technical\n",
                                encoding="utf-8")
            data = read_data(str(csv_path), text_column="body", label_column="department")
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["state"], "refund my invoice")
            self.assertEqual(data[0]["expected"]["label"], "billing")
            q = data[0]["questions"]["label"]
            self.assertEqual(q["type"], "choice")
            self.assertEqual(q["criteria"], ["billing", "technical"])

            # Test Excel UTF-8 BOM encoding
            bom_csv = Path(tmpdir) / "bom.csv"
            bom_csv.write_bytes("\ufeffbody,department\nrefund invoice,billing\ncrash,technical\n".encode("utf-8"))
            bom_data = read_data(str(bom_csv), text_column="body", label_column="department")
            self.assertEqual(len(bom_data), 2)
            self.assertEqual(bom_data[0]["state"], "refund invoice")

    def test_rows_from_csv_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "bad.csv"
            p.write_text("text,dept\nhi,billing\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                rows_from_csv(str(p), text_column="body", label_column="dept")
            p.write_text("text,dept\nhi,billing\nhello,billing\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                rows_from_csv(str(p), text_column="text", label_column="dept")

    def test_items_from_rows_supports_expected(self):
        exp_rows = [
            {"state": "please refund invoice", "questions": {"department": DEPARTMENT},
             "expected": {"department": "billing"}},
            {"state": "app crash", "questions": {"department": DEPARTMENT},
             "expected": {"department": "technical"}},
        ]
        items, skipped = items_from_rows(self.tok, exp_rows, 64, 40)
        self.assertEqual(len(items), 2)
        self.assertEqual(skipped, {})
        self.assertEqual(items[0]["target"], [1.0, 0.0, 0.0])
        self.assertEqual(items[1]["target"], [0.0, 1.0, 0.0])

    def test_items_from_rows_skip_tracking(self):
        exp_rows_with_empty = [
            {"state": "", "questions": {"department": DEPARTMENT}, "expected": {"department": "billing"}},
            {"state": "valid text", "questions": {"department": DEPARTMENT}, "expected": {"department": ""}},
            {"state": "valid text", "questions": {"department": DEPARTMENT}, "expected": {"department": "billing"}},
        ]
        items, skipped = items_from_rows(self.tok, exp_rows_with_empty, 64, 40)
        self.assertEqual(len(items), 1)
        self.assertEqual(skipped.get("empty_text"), 1)
        self.assertEqual(skipped.get("empty_label"), 1)

    def test_questions_are_rendered_exactly_as_inference_renders_them(self):
        # The sequence a training item produces must be the one `Agent` builds for the same
        # question, or the model is trained on option texts it never sees.
        from laya.agent import Agent

        questions = [
            DEPARTMENT, URGENT, LEVEL,
            {"type": "choice", "instructions": "which ?", "criteria": ["app", "screen"]},
            {"type": "noul", "instructions": "is it ?", "labels": {"false": "no", "true": "yes"},
             "criteria": {True: "holds", False: "does not hold"}},
            {"type": "noul", "instructions": {"which": "team"}},
        ]
        for question in questions:
            with self.subTest(question=question):
                k = len(render_options(Agent._to_internal(question)))
                item = self.item(question, [1.0 / k] * k)
                expected = build_sequence(self.tok, "app crash", Agent._to_internal(question), 64, 40)
                enc = encode_item(self.tok, item, 64, 40)
                self.assertEqual((enc["ids"], enc["markers"]), expected)

    def test_rows_share_one_tokenized_state(self):
        items, skipped = items_from_rows(self.tok, rows(1)[:1], 64, 40)
        self.assertEqual((len(items), skipped), (3, {}))
        self.assertTrue(items[0]["state_ids"] is items[1]["state_ids"] is items[2]["state_ids"])
        self.assertEqual([it["qtype"] for it in items], [QTYPES["choice"], QTYPES["noul"], QTYPES["score"]])

    def test_question_without_gold_is_not_an_item_or_a_skip(self):
        row = dict(rows(1)[0])
        row["gold"] = {"department": row["gold"]["department"]}
        items, skipped = items_from_rows(self.tok, [row], 64, 40)
        self.assertEqual((len(items), skipped), (1, {}))

    def test_bad_rows_are_skipped_with_a_reason(self):
        row = dict(rows(1)[0])
        row["questions"] = dict(row["questions"],
                                broken={"type": "choice", "instructions": "which ?", "criteria": []},
                                negative=DEPARTMENT)
        row["gold"] = dict(row["gold"], broken={"probabilities": {}},
                           negative={"probabilities": {"billing": -2.0}})
        items, skipped = items_from_rows(self.tok, [row], 64, 40)
        self.assertEqual((len(items), skipped), (3, {"invalid_question": 1, "invalid_target": 1}))

    def test_options_the_head_budget_collapses_are_skipped(self):
        # Two options that only differ after the per-option cap come out as the same span (#538).
        q = to_internal("q", {"type": "choice", "instructions": "which ?",
                              "criteria": {"one two three four five six": "", "one two three four five seven": ""}})
        state_ids = encode_state(self.tok, "app", 64)
        self.assertEqual(make_item(self.tok, q, [0.5, 0.5], state_ids, head_max_len=20),
                         (None, "options_collapsed"))
        self.assertIsNotNone(make_item(self.tok, q, [0.5, 0.5], state_ids, head_max_len=64)[0])

    def test_options_past_max_len_are_skipped_like_inference_refuses_them(self):
        # With a head longer than max_len, build_sequence drops the last options' markers and the
        # batch would crash on a target longer than its markers. Agent refuses the same question.
        from laya.common import build_head

        question = {"type": "choice", "instructions": "which ?",
                    "criteria": ["one", "two", "three", "four", "five", "six", "seven"]}
        q = to_internal("q", question)
        _ids, markers, _stats = build_head(self.tok, q, 64)
        state_ids = encode_state(self.tok, "app", 64)
        target = [1.0 / 7] * 7
        self.assertEqual(make_item(self.tok, q, target, state_ids, 64, max_len=markers[-1]),
                         (None, "options_beyond_max_len"))
        self.assertIsNotNone(make_item(self.tok, q, target, state_ids, 64, max_len=markers[-1] + 1)[0])
        self.assertIsNotNone(make_item(self.tok, q, target, state_ids, 64)[0])   # no max_len: not checked
        row = {"state": "app", "questions": {"q": question},
               "gold": {"q": {"probabilities": {"one": 1.0}}}}
        self.assertEqual(items_from_rows(self.tok, [row], markers[-1], 64), ([], {"options_beyond_max_len": 1}))
        # the same budget really does cut a marker in the sequence training would build
        _seq, kept = build_sequence(self.tok, "app", q, markers[-1], 64)
        self.assertEqual(len(kept), 6)

    def test_target_length_must_match_options(self):
        state_ids = encode_state(self.tok, "app", 64)
        self.assertEqual(make_item(self.tok, to_internal("q", DEPARTMENT), [0.5, 0.5], state_ids, 40),
                         (None, "target_mismatch"))

    def test_canonical_encoding_matches_build_sequence(self):
        item = self.item(DEPARTMENT, [0.8, 0.1, 0.1])
        enc = encode_item(self.tok, item, 64, 40)
        ids, markers = build_sequence(self.tok, "app crash", item["q"], 64, 40)
        self.assertEqual((enc["ids"], enc["markers"], enc["target"]), (ids, markers, [0.8, 0.1, 0.1]))

    def test_permuted_item_carries_a_permuted_target(self):
        item = self.item(DEPARTMENT, [0.7, 0.2, 0.1])
        order = [2, 0, 1]
        enc = encode_item(self.tok, item, 64, 40, option_order=order)
        self.assertEqual(enc["target"], [0.1, 0.7, 0.2])
        labels = list(DEPARTMENT["criteria"])
        # The token right after slot s's [MASK] is the label of option order[s].
        for slot, marker in enumerate(enc["markers"]):
            self.assertEqual(enc["ids"][marker], self.tok.mask_token_id)
            self.assertEqual(enc["ids"][marker + 1], self.tok.convert_tokens_to_ids(labels[order[slot]]))

    def test_permuted_training_input_is_what_inference_scores_for_that_order(self):
        # Inference with option_order builds the same tokens in the same slots, and
        # unpermute_probs maps the slot-ordered target back to the canonical one.
        from laya.agent import Agent

        order = [1, 2, 0]
        item = self.item(DEPARTMENT, [0.6, 0.3, 0.1])
        enc = encode_item(self.tok, item, 64, 40, option_order=order)
        internal = Agent._to_internal(dict(DEPARTMENT, option_order=order))
        expected_ids, _ = build_sequence(self.tok, "app crash", internal, 64, 40,
                                         option_order=internal["option_order"])
        self.assertEqual(enc["ids"], expected_ids)
        self.assertEqual(list(unpermute_probs(np.array(enc["target"]), order)), item["target"])

    def test_only_configured_types_are_shuffled(self):
        state_ids = encode_state(self.tok, "app", 64)
        choice, _ = make_item(self.tok, to_internal("q", DEPARTMENT), [1.0, 0.0, 0.0], state_ids, 40)
        noul, _ = make_item(self.tok, to_internal("q", URGENT), [0.5, 0.5], state_ids, 40)
        self.assertIsNone(draw_option_order(noul, random.Random(0), ("choice",)))
        orders = {tuple(draw_option_order(choice, random.Random(s), ("choice",))) for s in range(20)}
        self.assertTrue(all(sorted(o) == [0, 1, 2] for o in orders))
        self.assertGreater(len(orders), 1)
        self.assertEqual(draw_option_order(choice, random.Random(3), ("choice",)),
                         draw_option_order(choice, random.Random(3), ("choice",)))
        self.assertIsNone(draw_option_order(choice, random.Random(0), ()))

    def test_calibration_split_is_disjoint_and_seeded(self):
        items = list(range(50))
        train, calib = split_calibration(items, calib_max=400, calib_frac=0.1, seed=7)
        self.assertEqual((len(train), len(calib)), (45, 5))
        self.assertEqual(sorted(train + calib), items)
        self.assertEqual(split_calibration(items, 400, 0.1, 7), (train, calib))
        self.assertEqual(len(split_calibration(items, 3, 0.5, 7)[1]), 3)


class LossTests(unittest.TestCase):
    def test_soft_ce_ignores_padded_options(self):
        logits = torch.tensor([[2.0, 0.5, 9.0], [1.0, -1.0, 0.0]])
        target = torch.tensor([[0.75, 0.25, 0.0], [0.2, 0.3, 0.5]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        manual = -(torch.tensor([0.75, 0.25]) * torch.log_softmax(torch.tensor([2.0, 0.5]), -1)).sum()
        manual = (manual - (target[1] * torch.log_softmax(logits[1], -1)).sum()) / 2
        self.assertAlmostEqual(soft_ce_loss(logits, target, mask).item(), manual.item(), places=5)

    def test_rlcd_is_finite_and_differentiable(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3, requires_grad=True)
        target = torch.softmax(torch.randn(4, 3), -1)
        mask = torch.ones(4, 3, dtype=torch.bool)
        loss = rlcd_loss(logits, target, mask, torch.tensor([0, 1, 2, 0]), sigma=0.4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rlcd_matches_the_notebook_objective(self):
        # The loss as written in research/scripts/finetune_single_device.py, inlined: with the
        # same RNG state, rlcd_loss must give the same value and the same gradient.
        def script_loss(logits, target, mask, qtype, sigma):
            k = mask.sum(-1, keepdim=True).float()
            eps = torch.randn((4,) + logits.shape) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            ce = (target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            return -(adv * logp).mean() - ce

        torch.manual_seed(1)
        base = torch.randn(5, 4)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]],
                            dtype=torch.bool)
        target = torch.softmax(torch.randn(5, 4), -1) * mask
        target = target / target.sum(-1, keepdim=True)
        qtype = torch.tensor([0, 2, 1, 0, 2])
        results = []
        for fn in (lambda lg: script_loss(lg, target, mask, qtype, 0.3),
                   lambda lg: rlcd_loss(lg, target, mask, qtype, 0.3)):
            logits = base.masked_fill(~mask, -1e4).clone().requires_grad_(True)
            torch.manual_seed(5)
            loss = fn(logits)
            loss.backward()
            results.append((loss.item(), logits.grad.clone()))
        self.assertAlmostEqual(results[0][0], results[1][0], places=5)
        self.assertTrue(torch.allclose(results[0][1], results[1][1], atol=1e-6))

    def test_defaults_train_like_the_notebook(self):
        config = TrainConfig()
        self.assertEqual((config.loss, config.shuffle_options), ("rlcd", ()))
        self.assertEqual(config.micro_batch * config.grad_accum, 64)

    def test_config_rejects_unknown_values(self):
        for bad in (TrainConfig(loss="mse"), TrainConfig(shuffle_options=("choise",)),
                    TrainConfig(epochs=0), TrainConfig(calib_frac=1.0)):
            with self.assertRaises(ValueError):
                bad.validate()


class TrainingLoopTests(unittest.TestCase):
    def test_partial_accumulation_window_averages_its_own_steps(self):
        class ScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Identity()
                self.value = torch.nn.Parameter(torch.tensor(0.0))

        model = ScalarModel()
        tok = type("Tokenizer", (), {"pad_token_id": 0})()
        item = {"q": {"t": "choice"}, "k": 2}
        batch = {
            "marker_mask": torch.ones((1, 1), dtype=torch.bool),
            "target": torch.ones((1, 1)),
        }
        config = TrainConfig(
            epochs=1,
            micro_batch=1,
            grad_accum=2,
            head_lr=1.0,
            min_lr=1.0,
            weight_decay=0.0,
            grad_clip=100.0,
            loss="soft-ce",
            freeze_encoder=True,
            amp=False,
            log_every=0,
        )

        with patch("laya.train.encode_item", return_value={}), \
                patch("laya.train.collate_items", return_value=batch), \
                patch("laya.train._forward", side_effect=lambda current, *_args: current.value), \
                patch("laya.train.soft_ce_loss", side_effect=lambda logits, *_args: logits), \
                patch("laya.train.torch.optim.AdamW",
                      side_effect=lambda groups, weight_decay: torch.optim.SGD(
                          groups, weight_decay=weight_decay)):
            train_model(model, tok, [item] * 3, config, torch.device("cpu"), 1, 1)

        # The full two-step window and the final one-step window each contribute one update.
        self.assertAlmostEqual(model.value.item(), -2.0)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.tok = make_tokenizer()
        make_checkpoint(cls.root / "base", cls.tok)
        cls.data = cls.root / "train.jsonl"
        cls.data.write_text("\n".join(json.dumps(r) for r in rows()), encoding="utf-8")

    def run_finetune(self, out, **overrides):
        # Options stay in order here: a randomly initialised one-layer encoder cannot learn to read
        # option labels in a handful of epochs, only their slots. The permutation itself is
        # covered by DataTests; test_shuffled_run_saves_a_loadable_checkpoint runs it end to end.
        kwargs = dict(epochs=12, micro_batch=8, grad_accum=1, encoder_lr=1e-3, head_lr=1e-3,
                      calib_frac=0.2, log_every=0, loss="soft-ce", shuffle_options=())
        kwargs.update(overrides)
        config = TrainConfig(**kwargs)
        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")):
            return finetune(str(self.data), str(self.root / "base"), str(self.root / out), config, device="cpu")

    def test_finetune_learns_and_saves_a_loadable_checkpoint(self):
        report = self.run_finetune("out")
        self.assertEqual(report["train_items"] + report["calibration_items"], len(rows()) * 3)
        self.assertLess(report["epoch_loss"][-1], report["epoch_loss"][0])

        saved = json.loads((self.root / "out" / "rl_agent_config.json").read_text())
        self.assertTrue(saved["fine_tuned"])
        self.assertEqual((saved["max_len"], saved["head_max_len"]), (64, 40))
        # The inherited bucket would override the new fit at inference; too few items to refit it.
        self.assertNotIn("temperature_by_options", saved)
        self.assertTrue(all(TEMP_MIN <= t <= TEMP_MAX for t in saved["temperature"]))
        self.assertEqual(saved["training"]["laya_train"]["loss"], "soft-ce")
        self.assertTrue((self.root / "out" / "checkpoint_latest" / "model.safetensors").exists())

        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")):
            agent = load(str(self.root / "out"), device="cpu")
        answers = agent.predict("please refund invoice", {"department": DEPARTMENT})["answers"]
        self.assertEqual(answers["department"]["choice"], "billing")
        answers = agent.predict("app crash error", {"department": DEPARTMENT})["answers"]
        self.assertEqual(answers["department"]["choice"], "technical")

    def test_shuffled_run_saves_a_loadable_checkpoint(self):
        report = self.run_finetune("out_shuffled", shuffle_options=("choice",), epochs=1)
        saved = json.loads((self.root / "out_shuffled" / "rl_agent_config.json").read_text())
        self.assertEqual(saved["training"]["laya_train"]["shuffle_options"], ["choice"])
        self.assertEqual(len(report["epoch_loss"]), 1)
        agent = load(str(self.root / "out_shuffled"), device="cpu")
        self.assertIn(agent.predict("app", {"department": DEPARTMENT})["answers"]["department"]["choice"],
                      DEPARTMENT["criteria"])

    def test_rlcd_objective_still_trains(self):
        report = self.run_finetune("out_rlcd", loss="rlcd")
        self.assertLess(report["epoch_loss"][-1], report["epoch_loss"][0])

    def test_frozen_encoder_is_left_untouched(self):
        self.run_finetune("out_frozen", freeze_encoder=True, epochs=2)
        before = load_file(str(self.root / "base" / "model.safetensors"))
        after = load_file(str(self.root / "out_frozen" / "model.safetensors"))
        for name in before:
            if name.startswith("encoder."):
                self.assertTrue(torch.allclose(before[name].half().float(), after[name].float()), name)
        self.assertFalse(torch.allclose(before["scorer.1.weight"].half().float(), after["scorer.1.weight"].float()))

    def test_frozen_encoder_runs_without_dropout(self):
        from laya.train import load_checkpoint

        model, tok, _cfg = load_checkpoint(str(self.root / "base"))
        modes = []
        model.encoder.register_forward_pre_hook(lambda module, _args: modes.append(module.training))
        items, _ = items_from_rows(tok, rows(1)[:4], 64, 40)
        train_model(model, tok, items, TrainConfig(epochs=1, freeze_encoder=True, log_every=0),
                    torch.device("cpu"), 64, 40)
        self.assertTrue(modes)
        self.assertFalse(any(modes))

    def test_train_model_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            train_model(None, self.tok, [], TrainConfig(), torch.device("cpu"), 64, 40)

    def test_laya_train_cli_parser_and_validation(self):
        parser = build_train_parser()
        args = parser.parse_args(["--data", "train.csv", "--out", "./out", "--label-smoothing", "0.1",
                                 "--shuffle-options", "--freeze-encoder"])
        self.assertEqual(args.data, "train.csv")
        self.assertEqual(args.output_dir, "./out")
        self.assertEqual(args.label_smoothing, 0.1)
        self.assertTrue(args.shuffle_options)
        self.assertTrue(args.freeze_encoder)

        # --out is required when --dry-run is not set
        code_missing_out = train_cli_main(["--data", "train.csv"])
        self.assertEqual(code_missing_out, 2)

        # --dry-run without --out succeeds parsing
        dry_args = parser.parse_args(["--data", "train.csv", "--dry-run"])
        self.assertIsNone(dry_args.output_dir)
        self.assertTrue(dry_args.dry_run)

    def test_laya_train_cli_runs_on_csv(self):
        csv_path = self.root / "tickets.csv"
        lines = ["text,department"]
        for s in BILLING_STATES:
            lines.append(f"{s},billing")
        for s in TECH_STATES:
            lines.append(f"{s},technical")
        csv_path.write_text("\n".join(lines), encoding="utf-8")

        out_dir = self.root / "out_cli"
        code = train_cli_main([
            "--data", str(csv_path),
            "--base", str(self.root / "base"),
            "--out", str(out_dir),
            "--text-column", "text",
            "--label-column", "department",
            "--epochs", "1",
            "--micro-batch", "4",
            "--grad-accum", "1",
            "--loss", "soft-ce",
            "--label-smoothing", "0.05",
            "--device", "cpu",
        ])
        self.assertEqual(code, 0)
        self.assertTrue((out_dir / "model.safetensors").exists())
        self.assertTrue((out_dir / "rl_agent_config.json").exists())
        self.assertTrue((out_dir / "questions.json").exists())
        saved_q = json.loads((out_dir / "questions.json").read_text(encoding="utf-8"))
        self.assertIn("label", saved_q)
        self.assertEqual(saved_q["label"]["criteria"], ["billing", "technical"])

        agent = load(str(out_dir), device="cpu")
        ans = agent.predict("please refund invoice", {"label": {"type": "choice", "instructions": "Classify",
                                                               "criteria": ["billing", "technical"]}})
        self.assertIn(ans["answers"]["label"]["choice"], ["billing", "technical"])

        # Test --dry-run
        dry_code = train_cli_main([
            "--data", str(csv_path),
            "--base", str(self.root / "base"),
            "--text-column", "text",
            "--label-column", "department",
            "--dry-run",
        ])
        self.assertEqual(dry_code, 0)


class CalibrationReportTests(unittest.TestCase):
    @staticmethod
    def records(qtype, n, k=3, scale=1.0):
        # Logits that favour the target option by `scale`; target one-hot on option 0.
        return [(qtype, [scale] + [0.0] * (k - 1), [1.0] + [0.0] * (k - 1), k) for _ in range(n)]

    def test_each_kind_of_weak_fit_is_named(self):
        from laya.train import CALIB_WARN_N, calibration_report

        records = (self.records(QTYPES["choice"], 5) + self.records(QTYPES["noul"], 30, k=2)
                   + self.records(QTYPES["score"], CALIB_WARN_N + 10))
        report = calibration_report(records, [1.0, 1.0, 2.0])   # choice, score, noul
        self.assertEqual(report["choice"]["items"], 5)
        self.assertIn("not fitted", report["choice"]["issues"][0])
        self.assertEqual(len(report["noul"]["issues"]), 1)
        self.assertIn("only 30", report["noul"]["issues"][0])
        self.assertEqual(len(report["score"]["issues"]), 1)
        self.assertIn("unchanged at 1.0", report["score"]["issues"][0])
        self.assertAlmostEqual(report["score"]["accuracy"], 1.0)

    def test_clamped_and_healthy_fits(self):
        from laya.train import CALIB_WARN_N, calibration_report

        n = CALIB_WARN_N
        records = self.records(QTYPES["choice"], n) + self.records(QTYPES["score"], n)
        report = calibration_report(records, [TEMP_MIN, 1.3, 1.0])   # choice, score, noul
        self.assertIn("clamp", report["choice"]["issues"][0])
        self.assertEqual(report["score"]["issues"], [])
        # a type absent from the data is reported, but not as a problem
        self.assertEqual((report["noul"]["items"], report["noul"]["issues"]), (0, []))
        self.assertNotIn("accuracy", report["noul"])
        p = torch.softmax(torch.tensor([1.0, 0.0, 0.0]) / 1.3, -1)
        self.assertAlmostEqual(report["score"]["mean_confidence"], p.max().item(), places=6)

    def test_finetune_warns_and_records_a_weak_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_checkpoint(root / "base", make_tokenizer())
            config = TrainConfig(epochs=1, micro_batch=8, grad_accum=1, calib_frac=0.2, log_every=0)
            # The rows go straight to finetune through read_jsonl rather than through a file: nothing
            # here needs the JSONL parsing, which the end-to-end tests above already cover.
            # warnings.catch_warnings, not assertWarns: assertWarns walks every loaded module, and
            # transformers' lazy modules then try to import optional packages such as torchvision.
            with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")), \
                    patch("laya.train.read_jsonl", return_value=rows(2)), \
                    warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = finetune("rows.jsonl", str(root / "base"), str(root / "out"), config, device="cpu")
            saved = json.loads((root / "out" / "rl_agent_config.json").read_text())
        messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(any(re.match(r"laya\.train: \w+ calibration: not fitted", m) for m in messages), messages)
        self.assertEqual(saved["training"]["laya_train_calibration"], report["calibration"])
        self.assertEqual(sum(e["items"] for e in report["calibration"].values()), report["calibration_items"])


if __name__ == "__main__":
    unittest.main()
