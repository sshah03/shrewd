# shrewd

Turn LLM judgments into a small, fast, local text model for one fixed task.

I built shrewd while making local models for another project. It started with a question:
can GEPA prompt optimization get better labels from an LLM? Then, since Jev-style decisions
have gotten popular recently, I wanted to see how far I could get toward them locally for a
fixed set of questions. This repo contains the pipeline I used and what I measured along
the way.

It is useful when you repeatedly classify similar text, have a few hundred hand-labeled
examples and a larger unlabeled pool, and care about inference cost, latency, or keeping
text on your own hardware after training. It builds either a basic **classifier** (one label per
document) or a **decision panel** type of classifier (a fixed set of choices, yes/no probabilities,
and ratings answered together). Both use an LLM as the teacher and produce a saved model that runs
locally with no LLM in the loop.

Some findings may be useful even if you never use the library:

- Prompt optimization helped some teachers, but gains on the development split often
  disappeared on held-out data.
- On the two tasks tested for cost, choosing informative rows from a good teacher was
  more effective than buying cheaper labels or escalating uncertain ones.
- Better teacher labels did not always improve the student. Changing the student or
  giving it more training data sometimes helped more.
- Calibration helped the probabilities match human labels, but low calibration error
  sometimes hid a model that barely distinguished one example from another.

I used established methods. Most experiments weren't repeated with different data splits
and training seeds, so small score differences need more testing. I'm sharing the code
to make trying this on your own task easier. [BENCHMARKS.md](BENCHMARKS.md) has the
measurements, limits, and alternatives.

I'd especially like to hear which findings hold up on other people's data.

Four prebuilt decision panels (SMS spam, email triage, an AI-input guardrail, and a
personal-data gate) are available as demonstrations, with their known failures below.

```
seed CSV (labeled)    ──▶ [1] split: dev / locked test set
                          [2] optimize the teacher prompt on dev (GEPA)        optional
pool CSV (text only)  ──▶ [3] the teacher labels or judges the pool
                          [4] fix any of its answers by hand                   optional
                          [5] train a small local student on the result
                          [6] score teacher and student on the locked test set
```

**Part 1** covers using it. **Part 2** covers the findings and their limits, with fuller
tables in [BENCHMARKS.md](BENCHMARKS.md). The repo includes pipeline examples and panel
rebuilds. Some research results came from private experiment scripts that aren't included.

---

# Part 1: Using it

## Install

```
pip install "shrewd[teacher]"      # training: LLM client + prompt optimizer, tfidf student
pip install shrewd                 # inference only: sklearn stack, no LLM libraries
pip install "shrewd[teacher,embed]"     # + model2vec static-embedding student (no torch)
pip install "shrewd[teacher,setfit]"    # + SetFit student (torch, best few-shot)
pip install "shrewd[teacher,encoder]"   # + fine-tuned ModernBERT (torch, strongest at 1k+ rows)
```

## Compile a classifier

One label per document, a locked test set, and a report that says what to fix. Start here
if your problem is "which of these N buckets does this go in".

```python
import pandas as pd
from shrewd import Project

proj = Project(
    "runs/tickets",
    instructions="Classify customer support tickets by the customer's primary intent.",
    labels={"billing": "charges, invoices, refunds", "bug": "something is broken",
            "cancellation": "wants to cancel, pause, or downgrade", "other": "none of the above"},
    teacher="anthropic",   # or "openai", or any litellm model string
)
proj.add_seed(pd.read_csv("labeled.csv"))            # columns: text, label
proj.optimize(budget=800)                            # prompt optimization on the dev split
proj.label(pd.read_csv("unlabeled.csv"))             # dry_run=True prices it first
result = proj.distill(student="tfidf")
print(result.report())
```

```python
from shrewd import load

clf = load("runs/tickets")
clf.predict(["I was double charged last month"])   # -> ["billing"]
clf.predict_proba(["..."])                         # ndarray in clf.classes_ order
clf.predict(texts, min_confidence=0.52)            # None below the threshold: route to a person
```

To try it without an API key: `python examples/make_demo_data.py`, then
`python examples/quickstart.py`. It replays a bundled cache of a real run, so no API calls.

Options:

- **Label under a budget.** `proj.label(pool, budget_usd=20)` (or `n=2000`) acquires rows in
  rounds, picking the rows a quick tfidf probe is least sure about instead of going in file
  order. In my tests that took 1.1x to 2.3x fewer teacher calls for the same student
  accuracy. Re-running continues where it stopped.
- **Fix labels by hand.** After `label()`, `needs_review.csv` lists the rows the teachers
  disagreed on or could not answer, with an empty `human_label` column. Fill in the ones
  you care about and the next `distill()` uses them at confidence 1.0. Pass
  `teacher=[...]` with two or more models, or `votes=3`, to get disagreements to review.
  With one teacher and one vote the queue stays empty.
- **Pick a student.** `"tfidf"` (default, instant), `"embed"` (static embeddings), `"encoder"`
  (fine-tuned ModernBERT), `"setfit"` (contrastive fine-tuning), or your own object with
  `fit` / `predict` / `predict_proba` / `classes_` / `save`.

```python
rows = proj.compare(students=["tfidf", "embed", "encoder"])   # dev F1 vs latency vs size
proj.promote(rows[0]["student"])
result = proj.distill(student=rows[0]["student"])             # the one test-set evaluation
```

`autotune(dir, instructions, labels, seed_df, pool_df, budget_usd=...)` does this for you
under a spending cap. It starts with the cheapest teacher and moves up a tier only while
the dev score misses `target` and the next tier fits what's left of the budget. It scores
the locked test set once, on the winner, and writes every step to `autotune_trail.json`.

`teacher` takes any [litellm](https://docs.litellm.ai/docs/providers) model string.
`"anthropic"` and `"openai"` pick a strong default for that provider. **API keys come from
the environment**, the way litellm reads them: `export ANTHROPIC_API_KEY=...`,
`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, and so on. shrewd never takes a key as an argument or
writes one to disk, and `load()` doesn't need one.

`optimize()`, `label()` and `judge()` send your seed and pool text to the provider, so if
your text is sensitive, read their data-use terms first. The prompt goes out as a system
message marked for the provider's prompt cache (Claude charges ~10% of the input price for a
cached prefix), so a long prompt or many questions add little per document. Every response
is also cached locally in `cache.db`, so re-running a stage never pays twice.

**When to use something else:** if you won't run many documents through it, calling an
LLM or a hosted decision model may cost less than building and maintaining a student. Where
the line sits depends on your labeling, inference and retraining costs. It's also a poor fit
for tasks that need reasoning or fresh world knowledge per item, or for domains that change
quickly. The saved model only answers the task it was trained for.

## Compile a calibrated decision panel

The second path builds a fixed panel of typed questions (pick one of N, yes/no with a
probability, a rating on an ordered scale), all answered together in one pass. The
probabilities are calibrated against the teacher's answers and then scored against your
hand labels, because calibrating to the teacher alone doesn't guarantee
that a 0.8 means 80% on your task.

```python
import pandas as pd
from shrewd import Decisions, Choice, Noul, Score

d = Decisions(
    "runs/tickets",
    instructions="These are customer support tickets.",
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "charges, invoices, refunds",
                      "bug": "something is broken",
                      "other": "none of the above"}),
        "angry":    Noul(instructions="Does the customer sound angry?"),
        "severity": Score(instructions="How severe is this?",
                          criteria=["cosmetic", "workaround exists", "blocking"]),
    },
    teacher="anthropic/claude-fable-5-1",     # any litellm model string, or "anthropic" / "openai"
)
d.add_seed(pd.read_csv("labeled.csv"))    # text + one column per question id (blank where unknown)
d.judge(pd.read_csv("unlabeled.csv"))     # ONE teacher call answers every question per document
print(d.distill().report())               # trains, calibrates, scores against your hand labels
```

```python
from shrewd import load

dec = load("runs/tickets")
dec.decide("I've been charged twice and nobody will call me back")
# {"department": <Answer department choice='billing'>,
#  "angry":      <Answer angry noul=0.91>,
#  "severity":   <Answer severity score=1.4>}
dec.predict_proba(texts)                  # {question: (n x options) calibrated probabilities}
```

There are three question types. `Choice` picks one of up to 255 named options and returns
a probability for each. `Noul` (borrowed from Jev) answers a yes/no question with one
number, the probability of yes. `Score` rates against 2 to 10 ordered levels and returns
the probability-weighted mean, so a document split between "cosmetic" and "blocking" lands
in the middle. Describe each option in `criteria`. The teacher and the zero-shot stack read
those descriptions, and bare option names don't tell them much.

Options:

- **Fix the teacher's answers.** `judge()` writes `needs_review.csv` listing every
  (document, question) where the teacher's winning option is below 0.6, with an empty
  `human_answer` column. Write an option name, yes/no, or a level's number or description.
  the next `distill()` (or `apply_review()`) uses it as a certain target. Your answers are
  ledgered in `reviewed.csv` and survive any re-judging.
- **Optimize the prompt.** `d.optimize(budget=800)` runs GEPA over the free text above the
  questions, scored on your dev split by a proper scoring rule (it rewards accurate
  probabilities as well as accurate labels).
- **Choose the student.** `distill(features="auto")` fits tf-idf and static embeddings and
  keeps whichever wins on your hand-labeled dev split (0.01-0.5 ms/doc). `features="encoder"`
  fine-tunes ModernBERT (`pip install "shrewd[encoder]"`, 3-30 ms/doc, ~20-45 min of GPU
  or Apple-silicon training, ~11 GB RAM): use it when the question needs the model to
  read rather than count words. Part 2 covers where each one wins.
- **Add a zero-shot second opinion.** `distill(zero_shot=True)` stacks an NLI model onto
  each head and keeps it per question only where it beats the plain head on your hand-labeled
  dev split. It helped on sentiment and emotion and got rejected almost everywhere else.
  20-800 ms/doc.
- **Use several teachers.** `backend=EnsembleBackend(["anthropic/claude-fable-5-1",
  "openai/gpt-6-astra"])` averages their distributions. `backend="logprobs"` reads token
  probabilities where the API exposes them (OpenAI-compatible endpoints).

Adding a question later is cheap. Answers are cached per `(model, question, document)`, so
a ninth question reuses the eight you already paid for. Changing a question's text or
options throws away its cached answers. The compiled artifact answers exactly the questions it was
compiled for.

### Try a pre-built panel

Four decision panels for software that runs on a phone or laptop and would rather not
send text anywhere. Each one is a directory you load and call. I built them from public
data with Fable 5.1 as the teacher and scored them on a separate holdout that played no
part in training or prompt optimization. Only questions with public reference labels are
scored.

| panel | decides | questions with human labels | AUROC | ECE | ms/doc | student |
|---|---|---|---|---|---|---|
| `messages` | SMS spam / scam triage: unsolicited? · kind · asks_action? · risk | unsolicited | 0.997 | 0.002 | 6.3 | fine-tuned encoder |
| `email` | spam / phishing triage: spam? · phishing? · intent · impersonates? · urgency | spam | 0.993 | 0.010 | 0.5 | tf-idf |
| `guardrail` | what users type into an on-device AI feature: harmful? · jailbreak? · injection? · handling | harmful / jailbreak / injection | 0.937 / 0.971 / 0.987 | 0.018 / 0.013 / 0.056 | 0.03 | embeddings |
| `pii` | is this text safe to send off-device: contact · identity · financial · credentials · device · share_risk | all five | 0.941 / 0.921 / 0.963 / 0.965 / 0.995 | 0.005-0.079 | 9.4 | fine-tuned encoder |

```python
from shrewd import load

RELEASE = "https://github.com/sshah03/shrewd/releases/download/panels-v1/"
pii = load(RELEASE + "panel-pii.tar.gz")
pii.decide("my password is hunter2 and the PIN is 4471")
# {'contact': <Answer contact noul=0.01>, 'identity': <Answer identity noul=0.15>,
#  'financial': <Answer financial noul=0.03>, 'credentials': <Answer credentials noul=1.00>,
#  'device': <Answer device noul=0.00>, 'share_risk': <Answer share_risk choice='high'>}

sms = load(RELEASE + "panel-messages.tar.gz")
sms.decide("URGENT: your parcel is held. Pay the £2.99 fee at http://royal-mail-fee.co to release it")
# {'unsolicited': <Answer unsolicited noul=0.94>, 'kind': <Answer kind choice='scam'>,
#  'asks_action': <Answer asks_action noul=0.94>, 'risk': <Answer risk score=2.7>}
```

Those are real outputs on text none of the panels saw. They also show why `messages` ships
the encoder even though the tf-idf student tied it on the labeled test split. The SMS corpus
is from 2011, and on the parcel-payment scam above, the tf-idf head gives 0.64 and calls it
`personal`, while the encoder gets it right.

`pii` has a known gap. Its training sentences write card numbers as unbroken 16-digit runs,
so `4532 0151 1283 0366` with spaces or dashes scores 0.05 on `financial` (an IBAN, a wallet
address or an unbroken number scores 0.7-0.96). Adding spaced and dashed copies to the pool
fixed the dashed form (1.00), half fixed the spaced one (0.50), and made the unbroken one
worse (0.27), with the holdout score unchanged. More varied training examples might help,
but I haven't found a reliable fix.

These panels are examples of what this path produces, not models to adopt as is. Questions
with no public human labels (`kind`, `risk`, `intent`, `handling`, `share_risk`, etc.) were
only ever answered by the teacher, and ship that way.

- **Download.** The panels are attached to the
  [`panels-v1` release](https://github.com/sshah03/shrewd/releases/tag/panels-v1). `load()`
  takes the asset URL, downloads it once into `~/.cache/shrewd/`, and serves it from there.
  `messages` and `pii` (encoder) are ~550 MB and `email` and `guardrail` are under 30 MB.
- **Rebuild without a teacher.** `python examples/panels/build.py pii --from-judged`
  retrains from Fable 5.1's answers, which are committed under `examples/panels/judged/`.
  No API calls. It takes seconds for tf-idf or embeddings and ~25 min on a GPU for the
  encoder.
- **Adapt one.** `python examples/panels/fetch.py` pulls the data,
  `python examples/panels/build.py pii` builds it (1,500 pool-judging calls plus seed
  evaluation), and `pick.py` chooses the featurization using the seed test split as a
  validation set. The final scores come from the separate holdout. Question definitions are
  in `examples/panels/panels.py`. Copy one and change the questions to make your own.

## API

| | |
|---|---|
| `Project(dir, instructions=, labels=, teacher=, seed=42)` | create, or reload if `dir` has a manifest. `teacher` may be a list: each labels, the majority wins, confidence is the share that agreed |
| `.add_seed(df, test_frac=0.35)` | validate, split, and lock the test set |
| `.optimize(budget=800, target=None, reflection_model=None)` | optimize the teacher prompt on the dev split |
| `.label(df, votes=1, dry_run=False, concurrency=8, budget_usd=None, n=None)` | teacher-label the pool. `dry_run` prices it first. `budget_usd` or `n` switches to active acquisition |
| `.distill(student="tfidf", min_confidence=0.0, **kwargs)` | apply any human labels, train student, evaluate both, diagnose |
| `.apply_review()` | overrule the teacher wherever `human_label` is filled in `needs_review.csv`. ledgered so relabeling keeps it |
| `.compare(students=[...], min_confidence=0.0)` | rank candidates on dev F1, accuracy at 80% coverage, latency and size |
| `.promote(name)` | ship a compared candidate so that `load()` serves it |
| `load(dir)` | inference-only classifier: `predict(texts, min_confidence=0.0)` / `predict_proba` / `classes_` / `labels` |
| `autotune(dir, instructions, labels, seed_df, pool_df, budget_usd=, target=None, ...)` | the ladder above, automated under a spending cap |

**Calibrated decisions**

| | |
|---|---|
| `Decisions(dir, questions=, instructions=, teacher=, backend=, seed=42)` | create, or reload if `dir` has a manifest |
| `Choice(instructions=, criteria={option: description})` | pick one of N named options (max 255) |
| `Noul(instructions=, criteria={"true": ..., "false": ...})` | a yes/no question, answered with P(yes) |
| `Score(instructions=, criteria=[low, ..., high])` | 2-10 ordered levels, answered with the weighted mean |
| `.add_seed(df, test_frac=0.35)` | split hand-answered documents. `df` is text + one column per question id |
| `.optimize(budget=800, target=None, reflection_model=None, max_header_words=None)` | GEPA-optimize the prompt header on the dev split, objective `1 - Brier/2` (a proper scoring rule, not accuracy). `max_header_words` caps the header's length if cost is a concern. writes `prompt_header.txt` |
| `.judge(df, dry_run=False, concurrency=8)` | teacher answers every question per document in one call. writes `needs_review.csv` for its least confident answers |
| `.apply_review()` | overrule the teacher wherever `human_answer` is filled in `needs_review.csv`. ledgered in `reviewed.csv` |
| `.distill(features="auto", soft=True, calibration="auto", calibrate_teacher=False, zero_shot=False)` | train the heads on the teacher's distributions, calibrate them on the judged pool, score against your hand labels |
| `features=` | `"auto"` (fits tf-idf and static embeddings on the pool, keeps whichever wins on the hand-labeled dev split), `"tfidf"`, `"embed"`, `"encoder"` (fine-tuned ModernBERT, `[encoder]`), a factory, or any transformer with `fit_transform`/`transform` |
| `zero_shot=` | stack an off-the-shelf NLI model onto each head, kept per question only where it beats the plain head on the hand-labeled dev split (`[encoder]`) |
| `load(dir_or_url)` | `.decide(texts)` -> typed answers, `.predict_proba(texts, calibrated=True)`. an `https://...tar.gz` URL is downloaded once to `~/.cache/shrewd/` |

Backends: `"verbalized"` (default, works everywhere), `"logprobs"` (OpenAI-compatible
endpoints only), or `EnsembleBackend([...])` to average several teachers.

## Project directory

```
runs/tickets/
  manifest.json      task, seed, split hashes, per-stage cost log
  seed_dev.csv       optimization data
  seed_test.csv      LOCKED test set
  prompt.txt         best prompt, plain text
  optimize_log.json  every candidate the optimizer tried, with scores
  pool_labeled.csv   text, label, confidence (and round, when acquired under a budget)
  label_log.json     the learning curve: rows labeled, dollars spent, dev accuracy per round
  needs_review.csv   rows the teachers disagreed on or could not answer (fill in human_label)
  reviewed.csv       ledger of human labels, re-applied after every label()
  cache.db           LLM response cache (sqlite)
  student/           the trained model + meta.json
  report.json        metrics, intervals, findings
  candidates/        one dir per compare() candidate
  compare.json       the compare() leaderboard
```

## Development

```
uv venv && uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
pytest                                  # no API keys needed, optional model tests may download weights
ruff check .
```

An optional real-API smoke test runs with `SHREWD_E2E=1 pytest tests/test_e2e.py`.

---

# Part 2: What I measured

Every score below is against labels that didn't come from the teacher: human annotations,
issue tags, the product a consumer picked for their own complaint, and synthetic PII
annotations. Each table says which rows were scored. [BENCHMARKS.md](BENCHMARKS.md) has the
fuller methods and tables, and says what you can and can't rerun from this repo.

## What I tried

I started by optimizing a teacher prompt, using it to label a pool, and measuring how
much of the gain reached a small student. GEPA added about 6 points of teacher accuracy
with haiku on AG News (0.78 to 0.84), and 3 points with Sonnet 5 on Kubernetes, but only
with a stronger `reflection_model`. On 20 Newsgroups it added nothing. GEPA's own dev
score tends to overstate the gain, so check it against the test set. On a decision panel,
optimizing the prompt added about 0.03 teacher AUROC on GoEmotions and the student kept
roughly a third of that. Details are in
[BENCHMARKS.md](BENCHMARKS.md#prompt-optimization).

I then tried changing the teacher, student, training targets, and choice of rows to label.
The tables below cover what helped and what didn't. There are still things I haven't
tested, including changing GEPA's acceptance rule and feeding human review corrections
back into prompt optimization.

The four panels use the same pipeline on real tasks, but their data limits what they
show. The SMS corpus is from 2011, the PII sentences are synthetic, and several questions
only have teacher answers to train on. Their failures on newer scam texts and formatted
card numbers matter as much as their scores.

**Limits, and what isn't built.** Documents are plain strings. The structured
object/array input a hosted decision model accepts isn't supported. `Choice` stops at 255
options, and there's no two-stage narrow-then-choose for bigger sets. In practice it's
English only. A saved model won't answer a question it wasn't built for. The encoder
student needs ~11 GB of RAM to train. A student can pick up the teacher's systematic
mistakes.

## Results

Macro-F1 for the teacher and the best student, scored on the same rows. Full tables,
per-student latency and size, the confidence-signal study, and example commands are in
[BENCHMARKS.md](BENCHMARKS.md).

| dataset | classes | rows scored | teacher | best student | tfidf student |
|---|---|---|---|---|---|
| Banking77 | 6 | 126 test | 0.992 | 1.000 (setfit) | 0.992 |
| AG News | 4 | 84 test | 0.941 | 0.916 (setfit) | 0.833 |
| SMS Spam | 2 | 1,200 holdout | | 0.898 (tfidf) | 0.898 |
| SST-2 | 2 | 1,200 holdout | | 0.916 (setfit) | 0.677 |
| 20 Newsgroups | 5 | 1,200 holdout | | 0.825 (setfit) | 0.783 |
| Kubernetes issues | 5 | 2,659 holdout | 0.778 | 0.718 (tfidf) | 0.718 |
| CFPB complaints | 6 | 1,309 holdout | 0.782 | 0.760 (setfit) | 0.737 |

The teacher is Claude Sonnet 5 throughout. Kubernetes and CFPB use organic labels
(maintainer tags, the consumer's own product choice), and that's where the teacher itself
becomes the limit. The CFPB students were frozen after training on H1-2025 complaints and
lost nothing when scored on complaints from 6 and 12 months later.

The student's confidence picks out rows where it's more accurate. On its most confident
rows, its accuracy can match or beat the teacher's accuracy on all rows. That doesn't mean
it matches the teacher on those same rows. The table shows student accuracy when keeping
only its most confident share of rows. The teacher column is its accuracy on all rows, for
reference:

| dataset | teacher, all rows | student | 100% | 80% | 70% |
|---|---|---|---|---|---|
| CFPB complaints | 0.811 | tfidf | 0.759 | 0.810 | 0.835 |
| CFPB complaints | 0.811 | setfit | 0.784 | 0.877 | 0.901 |
| Kubernetes issues | 0.782 | tfidf | 0.733 | 0.794 | 0.828 |
| 20 Newsgroups | 0.885* | setfit | 0.824 | 0.901 | 0.923 |

\* teacher agreement with labels on the pool. The report prints this curve with the
threshold for each coverage level, and `clf.predict(texts, min_confidence=0.52)` returns
`None` below it so you can send those rows to the teacher or a person. I read these
thresholds off the evaluation data. For real use, pick the threshold on validation data,
then check both models on the rows you keep, and the whole setup with its fallback on a
fresh holdout.

If you already run an LLM classifier in production and want to keep the LLM as a fallback,
look at [TRACER](https://github.com/adrida/tracer). There's a comparison with it and with
DSPy, Autolabel, Cleanlab and the annotation tools in
[BENCHMARKS.md](BENCHMARKS.md#compared-to-other-tools).

## What the pipeline guards

- **The test set is locked.** `add_seed` writes it once and records a hash. The optimizer
  never sees it, `compare()` ranks students on the dev split, and only `distill()` scores
  it, for teacher and student together. Pool rows that duplicate seed texts are dropped
  from training. Every macro-F1 comes with a bootstrap interval, and the student-vs-teacher
  gap is tested with a paired interval before the report calls it a problem. If you keep
  choosing configurations by their test scores, the test split turns into a validation
  set. Use a separate holdout for the final numbers, as the panel examples do.
- **The diagnosis is a fixed set of rules.** The report tells you which problem you have:
  teacher too weak, student too small, an ambiguously defined label (with the confused pair
  and real examples), starving classes, or a seed-vs-pool distribution shift.
- **Everything is on disk and resumes.** Plain CSV and JSON files, with one sqlite cache in
  front of every teacher call. Kill `label()` halfway and rerun it. Finished stages make no
  API calls.

`prompt.txt` is plain text. Edit it, or skip `optimize()` and write your own. `label()`
uses whatever is there.

## Calibrated decisions: how it works and what each piece is worth

Asking all the questions in one call keeps the cost down. The document is the expensive
part of the prompt, so asking eight questions separately pays for it eight times. One call
per document made judging a 1,500-comment pool about eight times cheaper. The student works
the same way (one featurization, one linear head per question), so answering eight
questions about a document takes 0.044 ms on one CPU core.

### Calibration

Calibration is fit against the teacher's answers and checked against human labels.
`distill()` fits a calibrator per question on out-of-fold predictions over the judged pool,
using the teacher's most likely answer as the target (or a human correction where supplied).
This costs no extra API calls. Cross-fitting keeps each calibration prediction apart from
its training rows, but the teacher's systematic mistakes can still carry through. On 4,000
held-out GoEmotions comments, across eight questions with base rates from 2.5% to 25%,
calibration also brought the probabilities closer to what human raters said:

| | accuracy | ECE | Brier | AUROC | \|p - rater rate\| |
|---|---|---|---|---|---|
| raw | 0.904 | 0.122 | 0.090 | 0.765 | 0.208 |
| calibrated | 0.931 | 0.043 | 0.058 | 0.765 | 0.114 |

Accuracy goes up because a yes/no answer calibrated below 0.5 flips, which is right when
the model was running high across the board. The same student beats the Claude teacher it
learned from on ECE (0.043 vs 0.099) and Brier, and runs 3,755x faster for free, though the
teacher still ranks documents better. `report()` also catches a failure that a good ECE can
hide. A head that answers the base rate on every document is perfectly calibrated and
useless, and the report calls it a failure.

### Calibration vs ranking

Good calibration doesn't mean good ranking. On GoEmotions, calibrated setups had similar
ECE but ranked documents quite differently, and both the featurization and the training
targets changed the ranking. Across the datasets in BENCHMARKS, static embeddings beat
tf-idf on short informal text and lose to it on long domain text. The fine-tuned encoder is
the best student wherever word counts fall short (AG News +1.9, 20 Newsgroups +1.3, CFPB
+0.5, Kubernetes +4.7 over tf-idf, SST-5 a third of a level, and the pii panel's `identity`
question 0.739 to 0.921), at 3-30 ms/doc. `soft=True` (training on the teacher's full
distribution instead of just its top answer) is worth 1-2 points of AUROC and cuts raw ECE
tenfold. It's the default.

### Zero-shot stack

The stack only helps where the base model is weak and the NLI model is strong. Its
hypotheses are built from your option descriptions. Each one gets a learned
bias so a broad option can't soak up probability from narrow ones, and a gate keeps the
stack per question only where it beats the plain head on the hand-labeled dev split. The
gate uses hand labels because on convention-driven labels the zero-shot model and the
teacher make the same mistakes (77% of the time on Kubernetes), so agreeing with the
teacher makes the stack look better than it is. It took GoEmotions over tf-idf from 0.755
to 0.836 and SST-2 to 0.97 over either featurizer. On every multi-class set and all twelve
human-labeled questions in the four panels, it was either rejected by the gate or no better
than the best plain student. 20-800 ms/doc.

### Teacher choice

A stronger teacher helped on this panel. Same panel and pool with Fable 5.1 instead of
Sonnet 5: teacher AUROC went from 0.838 to 0.868, the student from 0.785 to 0.803 (tf-idf)
and 0.853 to 0.858 (stacked), with lower ECE on all of them, for 2.2x the judging cost. A
teacher that writes out its probabilities only uses about 20 distinct values per question,
and AUROC gives each tied yes/no pair half credit. If you broke those ties using the true
labels, Sonnet 5 would go from 0.838 to 0.870. That's a best case and not something the
teacher can actually do. Its measured AUROC is still 0.838.

### Prompt optimization

On this panel the gain held up on unseen data. GEPA on the prompt header, scored by
`1 - Brier/2` on dev, took teacher AUROC from 0.838 to 0.864 and cut Brier by 12% on
labeled rows it never saw. The student kept a third of the ranking gain
and all of the calibration gain. The 767-word header it wrote made each teacher call cost
2.4x as much. Once the system prompt was marked for the provider's cache that dropped to
1.8x, and the rest comes from the teacher writing more reasoning. There's
no cap by default. Set `max_header_words` if the per-document cost matters.

### Inference

Inference runs a saved local model. In my checks, the tf-idf/embedding panel gave
identical outputs across repeated calls, save and load, batch sizes, and input order.
Nothing is sampled and no hosted model changes under you. Encoder results can differ very
slightly across hardware or batch sizes. The tf-idf/embedding panel answered 4,000
documents x 8 questions in 130 ms on one core, about 246,000 answers a second.

### What didn't work

- A ranked-probability term in the encoder's loss. It steadies training without gradient
  clipping and does nothing once you clip.
- Policy-gradient training on a differentiable proper-scoring objective. More variance, no
  improvement.
- Calibrating on the small hand-labeled dev split instead of the judged pool. Too few rows,
  and the result was worse.
- Static embeddings as the default featurizer. They lose 8 to 11 points on long domain
  text, which is why `auto` measures both.
- Gating the stack on agreement with the teacher on the pool. That said +4 where hand
  labels said -0.3.

Details are in [BENCHMARKS.md](BENCHMARKS.md#what-does-not-work).

### Comparison with Laya

I trained and calibrated the students on each task, while Laya
answered zero-shot with no extra training or calibration. The students scored better on
six of eight datasets and had lower ECE on all eight. Laya led on AG News and GoEmotions
(0.907 vs 0.858 here on the latter). Training data, architecture and calibration all
differ between the two, so this doesn't tell you which architecture is better. The
100-25,000x speedups are for the tf-idf and embedding students. Full numbers and how I
prompted Laya are in [BENCHMARKS.md](BENCHMARKS.md#every-student-on-every-dataset).

## Labeling under a budget

Labeling a pool in file order spends most of the money on rows the student would get
right anyway. Pass a budget or a row cap and `label()` works in rounds instead. A tfidf
probe trained on what's labeled so far picks the rows it's least sure about, spread across
clusters, and the teacher labels that batch. The probe's accuracy on the hand-labeled dev
split is logged each round so you can watch the curve flatten.

```python
proj.label(pool, budget_usd=20)     # or n=2000, re-running continues where it stopped
```

Teacher calls needed for the tfidf student to get within one point of its full-pool
accuracy, labeling in file order versus picking by uncertainty, scored on labeled holdouts:

| dataset | pool | in order | acquired | saving |
|---|---|---|---|---|
| AG News | 5,000 | 2,912 | 1,248 | 2.3x |
| Kubernetes issues | 1,250 | 1,144 | 520 | 2.2x |
| CFPB complaints | 1,500 | 1,000 | 500 | 2.0x |
| SST-2 | 500 | 451 | 328 | 1.4x |
| 20 Newsgroups | 1,250 | 1,040 | 936 | 1.1x |

This was a simulation on pools that were already labeled. A separate cost study found
that Sonnet with this kind of selection hit its target for a third of the cost on CFPB and
half on AG News. Cheaper haiku labels, and the escalation schemes I tried, didn't beat that
at the prices I measured. Rows picked by the tfidf probe were about as useful to the
embedding student as rows in random order, so using the probe doesn't lock you into
shipping tfidf. A pool picked this way leans toward hard rows on purpose, so the
seed-vs-pool distribution check is skipped for it.