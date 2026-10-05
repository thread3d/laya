# laya-java

JVM inference for Laya: `Agent.predict` / `Agent.predictBatch` over the BPE tokenizer, the sequence
builder, ONNX Runtime and typed answers. JDK 17+. One dependency — `com.microsoft.onnxruntime`.

Laya is a non-autoregressive "System 1" decision model: a bidirectional encoder plus a typed head
that answers `choice` / `score` / `noul` questions about a state in **one forward pass**, with
calibrated confidences. Every question about the same state becomes one row of a single batch, so
four questions about a document cost one batched encode rather than four round trips.

## Status

Implemented: tokenizer, sequence builder, config, ONNX inference (fused or split graph), answer
decoding, `predict`, `predictBatch`, usage and truncation reporting.

Not implemented yet: hooks, the language `Router`, `predictLong`, shortlist, structured `decide`,
the `laya-java-client` HTTP module, Android. **Not published to Maven Central** — see
[Installing](#installing).

## Export a graph (once per checkpoint)

```bash
python scripts/export_onnx.py --model <ckpt-dir-or-hub-id> --output ./model/laya.onnx
```

That writes the fused graph. The split form (`encoder.onnx` + `head.onnx`, from
`laya-ts/scripts/export_onnx.py`) is also supported; `LayaSession` opens whichever a directory
holds, preferring the fused one.

A checkpoint directory must contain `rl_agent_config.json` and `tokenizer/` — that is where the
budgets, the fitted temperatures and the special tokens live, and none of them may be defaulted.
The two shipped checkpoints disagree on every one: english is `max_len 512 / head_max_len 192` with
fitted per-bucket temperatures, multilingual is `1024 / 256` with none, and multilingual reuses
`<bos>` as its classification token and `<eos>` as its separator.

## Ask questions

```java
import com.convaiinnovations.laya.*;
import java.nio.file.Path;
import java.util.*;

// Options are POSITIONAL, so use a LinkedHashMap: two orders are two different questions.
Map<String, Object> criteria = new LinkedHashMap<>();
criteria.put("refund", "money back for a duplicate charge");
criteria.put("escalate", "pass it to a human");
criteria.put("ignore", "no action needed");

Map<String, Question> questions = new LinkedHashMap<>();
questions.put("intent", Question.choice("What does the customer want?", criteria));
questions.put("urgent", Question.noul("This needs a human today."));
questions.put("severity", Question.score("Rate the severity.",
        List.of("none", "minor", "major", "critical")));

try (Agent agent = Agent.open(Path.of("./checkpoint"), Path.of("./model"))) {
    Prediction p = agent.predict("We were billed twice for March and want a refund today.",
            questions, "en");   // the language tag may select a temperature override

    Answer.Choice intent = (Answer.Choice) p.answer("intent");
    System.out.println(intent.choice());                 // "refund"
    System.out.println(intent.probabilities());          // {refund=..., escalate=..., ignore=...}
    System.out.println(intent.answerConfidence());       // max(p), the calibrated one

    Answer.Noul urgent = (Answer.Noul) p.answer("urgent");
    System.out.println(urgent.noul());                   // P(the statement holds)

    Answer.Score severity = (Answer.Score) p.answer("severity");
    System.out.println(severity.score());                // expected value over level indices
    System.out.println(severity.legend());               // {0=none, 1=minor, 2=major, 3=critical}
}
```

`Answer` is sealed: the three shapes are a closed set, so a reader knows there is no fourth, and an
`instanceof` pattern needs no `else` that throws.

```java
Answer a = p.answer("intent");
if (a instanceof Answer.Choice c) {
    handle(c.choice(), c.probabilities());
} else if (a instanceof Answer.Score sc) {
    handle(sc.score(), sc.legend());
} else if (a instanceof Answer.Noul n) {
    handle(n.noul());
}
```

On **JDK 21 or newer** a pattern `switch` over it is exhaustive and a missing case is a compile
error. That is a language feature of 21, not of this library — on JDK 17, which this module
targets, pattern switches are a preview feature and the `instanceof` form above is the portable
one.

### Two confidences, deliberately

`answerConfidence()` is the probability mass on the reported answer — `max(p)` — and it is the
quantity temperature scaling fits and every calibration figure in laya is computed on.
`confidence()` means something different per type: normalised entropy for a choice or a score, and
`max(p, 1-p)` for a noul. **They are on different scales and must not be compared against the same
threshold.**

The calibration guarantee on `answerConfidence` is conditional, and the condition is not met by
default: it holds only after the temperatures have been fitted and validated on held-out data for
that checkpoint and option count. The shipped checkpoints are over-confident.

## Usage, and what was silently dropped

A state longer than the room left after the question's head is truncated **without an error**, and a
caller cannot reconstruct that from outside: the budget is in tokens, not characters, and the room
left moves per checkpoint and per question.

```java
Usage u = p.usage();
u.inputTokens();            // non-padding tokens fed to the encoder, across every row
u.stateTokens();            // what the state serialised to, before any budget was applied
u.stateTokensDropped();     // the worst case across this state's questions
u.truncated();              // whether any question dropped state tokens
u.truncatedQuestions();     // which ones
u.collapsedOptions();       // questions whose options no longer have a token span each
```

`collapsedOptions()` is the one a reviewer of your prompts will care about: two options whose
rendered text is identical for the first 48 tokens collapse to the same span, the model cannot tell
them apart, and **nothing else downstream can see it** because the marker count still matches. It
reports the count the question *defines*, not the number of markers that reached the sequence.

A question whose option markers do not all fit `max_len` is **refused**, not answered partially —
answering would return a distribution over whichever options survived, with the rest absent from the
answer and unchoosable.

## Many states, one call

```java
List<Object> states = List.of(emailA, emailB, emailC);
List<Prediction> out = agent.predictBatch(states, questions);        // all in one graph call
List<Prediction> out2 = agent.predictBatch(states, questions, "en",
        /* batchSize */ 16, /* sortByLength */ true);                // bounded memory, less padding
```

Results come back in the caller's order whatever the grouping was. `batchSize` bounds peak memory
and `sortByLength` cuts padding; neither changes an answer.

## Lower-level pieces

```java
Tokenizer tok = Tokenizer.fromModelDirectory(Path.of("./checkpoint"));
int[] ids = tok.encode("charged twice");
int[] capped = tok.encode(longText, 48);          // stops early; same prefix as the full encoding
AgentConfig cfg = AgentConfig.fromModelDirectory(Path.of("./checkpoint"));
```

`Agent.using(tokenizer, config, session)` assembles an agent from parts — for a caller that already
holds them, or to drive the batching and usage accounting through a stub
`infer.InferenceSession` instead of a 1.2 GB graph.

## Threads

`Agent.open(model, graph)` leaves the thread count to ONNX Runtime, which is what the Python
runtime does. Pass `Agent.open(model, graph, 1)` for a request-per-thread server, where the
parallelism is already in the requests. One `Agent` is **not** safe for concurrent `predict` calls
unless ONNX Runtime is configured for it; hold one per worker, or serialise access.

## Build and test

```bash
cd laya-java
./gradlew build                 # -Xlint:all -Werror
./gradlew test                  # the suite that needs no checkpoint
```

The parity tests need a checkpoint and a graph, and **abort with an actionable message** without
them rather than passing vacuously:

```bash
LAYA_CHECKPOINTS=/path/to/checkpoints \
LAYA_ONNX_GRAPH=/path/to/model/laya.onnx \
  ./gradlew test
```

## Parity: generated, not asserted

`laya-java/fixtures/*.json` are generated **from** the Python package and committed. The drift check
is what keeps them honest:

```bash
python laya-java/scripts/gen_fixtures.py            # regenerate
python laya-java/scripts/gen_fixtures.py --check     # fail if a committed fixture would change
```

A family that needs a checkpoint or a graph this run cannot reach is **left alone** and reported as
unverified, so running the generator without the checkpoints cannot replace good expectations with
skip markers. `--strict` turns "could not verify" into an error, for a lane that is supposed to have
them.

Where it does *not* match Python bit for bit, and why: the ONNX logits are float32 and Python
computes the softmax in float32, while this port — like `laya-ts` and `laya-dotnet` — computes it in
`double`. Over 200,000 random logit rows the four-decimal probabilities differ on 88 of them.
Matching float32 exactly would mean reproducing NumPy's own float32 `exp` (which differs from
rounding a double `exp` on 0.65% of values) and its pairwise summation order; neither is a contract
NumPy publishes. Rounding, by contrast, *is* exact: `BigDecimal` at scale 4 with `HALF_EVEN` is
Python's `round(v, 4)` on Python's operand.

## Installing

**Not published yet.** Build from source into your local repository:

```bash
cd laya-java && ./gradlew publishToMavenLocal
```

```kotlin
dependencies { implementation("com.convaiinnovations:laya-java:0.1.0-SNAPSHOT") }
```

Publishing to Maven Central needs the `com.convaiinnovations` namespace verified by the project
owner, a Central portal token and a PGP signing key. `.github/workflows/release-java.yml` is wired
for it and has never been run; signing activates only when a key is present, so `build` and
`publishToMavenLocal` work without any of it.
