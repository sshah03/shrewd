# Measured results

These are the experiments I ran with this repo. The repo includes examples for training
classifiers, scoring saved students, and rebuilding the four panels from cached teacher
answers. Some experiments used scripts I'm keeping private: the active-selection
simulations, cost curves, GEPA variants, and some ablations and comparisons. Their results
are written up here, but you can't reproduce every table from this repo alone.

`python examples/benchmark.py [agnews|banking77|sms|sst2|newsgroups]` pulls a public labeled
dataset, hides most labels to form an unlabeled pool, runs the pipeline, and scores the
teacher's bulk labels against the hidden gold. Its defaults are a starting point, not the
exact setup of every run below. `examples/score_holdout.py <project_dir> <labeled csv>`
scores a saved classifier. The panel fetch/build scripts and committed answers support
panel rebuilds with `--from-judged`, and the README describes that workflow. Fresh teacher
calls, changed data sources, and dependency versions can change results.

## How to read the tables

The locked test sets are small (42 to 126 rows) because they are carved from the same
few-hundred-row seed a real user would have. That's the whole premise of the library, so
wherever more gold was available the tables also have a holdout column: the same saved
students scored on 1,200 or more held-out rows. When the two disagree, trust the holdout.
The report prints a bootstrap interval next to every macro-F1 for the same reason.

Most experiments weren't repeated with different data splits and training seeds, so small
score differences need more testing. The reference labels come from human annotations,
organic labels such as issue tags, and synthetic PII data.
For the four panels, I used the seed test split to choose a model, making it a validation
set. The scores in their table come from separate holdouts. I also inspected holdout scores
across variants during the experiments, so those holdouts helped inform later decisions.

**Pool labels vs gold** is the other number to watch. It is measured on the full hidden
pool (500 to 5,000 rows per dataset) and tells you what actually went into the student's
training data.

The classifier datasets right below use Claude Sonnet 5 as the teacher, one vote per item.
Later sections name their own teachers. Classifier students train on the teacher-labeled
pool plus the gold dev split, which is exactly what `distill()` does.

## Public datasets

**Banking77, 6 intents** (360 seed / 411 pool / 126-row locked test)

| | macro-F1 |
|---|---|
| Teacher (pool labels 98.3% correct) | 0.992 |
| setfit student | 1.000 |
| tfidf student | 0.992 |

**AG News, 4 topics** (240 seed / 5,000 pool / 84-row locked test)

| | macro-F1 | ms/item | size |
|---|---|---|---|
| Teacher (pool labels 92.2% correct) | 0.941 | ~700 | API |
| setfit student | 0.916 | ~10 | 420MB |
| encoder student | 0.904 | 12.5 | 602MB |
| embed student | 0.845 | 0.03 | 59MB* |
| tfidf student | 0.833 | 0.13 | 13.6MB |

Trained on the 156 gold seed rows alone, the same students score 0.735 (tfidf) and 0.808
(embed). The teacher-labeled pool closes the gap.

**SMS Spam, 2 classes** (120 seed / 500 pool / 42-row locked test)

| | test macro-F1 | 1,200-row holdout |
|---|---|---|
| Teacher (pool labels 99.4% correct) | 1.000 | |
| tfidf student | 1.000 | 0.898 (96.8% acc) |

This is the first imbalanced set (87% ham). The 42-row test says the sub-1MB student is
perfect, and 1,200 held-out rows say 0.898. The misses were in the minority class.

**SST-2 sentiment, 2 classes** (120 seed / 500 pool / 42-row locked test)

| | test macro-F1 | 1,200-row holdout | ms/item | size |
|---|---|---|---|---|
| Teacher (pool labels 97.0% correct) | 0.976 | | ~700 | API |
| setfit student | 0.881 | 0.916 | 4.9 | 439MB |
| encoder student | 0.905 | 0.848 | 3.8 | 602MB |
| embed student | 0.832 | 0.730 | 0.02 | 59MB* |
| tfidf student | 0.808 | 0.677 | 0.08 | 0.9MB |

Sentiment depends on negation and phrasing that n-grams can't see. The 42-row test ranked
the encoder first. On 1,200 held-out rows setfit comes first by 7 points and tfidf drops
by 13.

**20 Newsgroups, 5 groups** (300 seed / 1,250 pool / 105-row locked test)

| | test macro-F1 | 1,200-row holdout | ms/item | size |
|---|---|---|---|---|
| Teacher (pool labels 88.5% correct) | 0.904 | | ~700 | API |
| setfit student | 0.826 | 0.825 | 11.5 | 439MB |
| encoder student | 0.789 | 0.807 | 22.4 | 602MB |
| tfidf student | 0.808 | 0.783 | 0.30 | 12.4MB |
| embed student | 0.703 | 0.744 | 0.07 | 59MB* |

The hardest public set, with noisy Usenet posts and overlapping topics. The setfit student
used `max_seq_length=256` to keep pair training within a laptop's memory.

\* The embed student saves only its 0.03MB logistic head. The 59MB potion-base-8M embedder
is fetched from Hugging Face on first `load()`. It is the one student whose inference needs
network access once.

## Kubernetes issue triage (organic labels)

4,209 kubernetes/kubernetes issues, 5-way `kind/*` classification, maintainer labels as
gold. 300 seed / 1,250 pool / 2,659-row holdout. Teacher and student
scored on the same 2,659 held-out issues:

| | macro-F1 | agreement with maintainers | serving |
|---|---|---|---|
| Teacher (Sonnet 5) | 0.778 | 78.2% | one API call per issue |
| tfidf student | 0.718 | 73.3% | local, 0.4 ms/issue, 19MB |

Organic labels are harder than benchmarks. Sonnet 5 agreed with maintainers on 78% of the
pool, and the diagnosis flagged the teacher as the bottleneck. Reading the disagreements, it
looked like the k8s labels record how the issue was resolved ("this bug report is
`kind/support`, it was your config"), which often isn't in the issue text. Spelling that
out in the label descriptions moved agreement from 78.0% to 78.6%, which is noise. A
stronger teacher helped more. Fable 5 with the same prompt reached 81.4% at about 4x the
labeling cost, and the two teachers agree with each other on 90%. But the student scored
the same from either teacher's labels (0.718 vs 0.717 on the holdout). At this data size
the better labels didn't seem to reach the student.

Reproduce on any repo: `python examples/fetch_github_issues.py owner/repo label=class ...`,
then the quickstart, then `python examples/score_holdout.py <project_dir> <labeled csv>`.

## CFPB complaint routing, with a 12-month drift test

Real, timestamped consumer complaint narratives routed to 6 financial products, with the
consumer-selected product as gold. 300 seed / 1,500 pool drawn from H1-2025. The students
were frozen after training on H1-2025 data, then scored as is on complaints from 6 and 12
months later. Macro-F1 against the organic labels, with the teacher scored on the same rows:

| | H1-2025 (1,309) | +6 mo (1,020) | +12 mo (1,020) | ms/item | size |
|---|---|---|---|---|---|
| Teacher (Sonnet 5) | 0.782 | 0.841 | 0.847 | ~900 | API |
| setfit student | 0.760 | 0.803 | 0.817 | 14.5 | 439MB |
| tfidf student | 0.737 | 0.793 | 0.789 | 0.35 | 9.8MB |
| encoder student | 0.733 | 0.775 | 0.797 | 35.6 | 602MB |
| embed student | 0.659 | 0.721 | 0.718 | 0.08 | 59MB* |

I saw no drop a year out. Every student's gap to the teacher is about as small at +12
months as on day one (setfit went from 2.2 points to 3.0, inside the roughly 2.4-point 95%
interval each window has). Scores move between windows in step with the teacher's own
scores, so it's the data mix changing and not the model getting worse.

Trained on the 195 gold dev rows alone, the same students score 0.732 (setfit), 0.651
(tfidf), 0.573 (encoder) and 0.542 (embed) on the base window. The teacher-labeled pool is
worth 3 points for setfit, which is built for few-shot, and 9 to 16 for the rest, widening
to 6 to 24 on the +12-month window.

At scale, routing 1M complaints a year through the teacher costs about $3,400 and about a
second per item. On my laptop CPU the setfit student costs nothing per item at 14.5 ms, and
tfidf takes 0.35 ms in under 10MB. After the build, no complaint text leaves your machines.

Reproduce: `python examples/fetch_cfpb.py`, then the quickstart's five calls on
`base.csv`'s seed/pool split, then `examples/score_holdout.py` against each window.

## Active acquisition of pool rows

Simulated on the already-labeled pools, with the teacher's labels standing in for new
teacher calls. Start from the gold dev split, add pool rows in twelve equal batches under
each strategy, retrain, and score on the gold holdout. "Margin" picks the rows whose top two
probabilities are closest. "Margin+diverse" takes the 4x most uncertain, clusters them and
keeps one per cluster.

Labels needed to reach the full-pool student's accuracy minus one point:

| dataset / student | full-pool acc | random | margin | margin+diverse |
|---|---|---|---|---|
| AG News (5,000) / tfidf | 0.877 | 2,912 | 1,248 | 1,248 |
| AG News / embed | 0.860 | 416 | 832 | 832 |
| 20 Newsgroups (1,250) / tfidf | 0.782 | 1,040 | 832 | 936 |
| 20 Newsgroups / embed | 0.746 | 936 | 520 | 728 |
| CFPB (1,500) / tfidf | 0.759 | 1,000 | 500 | 500 |
| CFPB / embed | 0.686 | 1,125 | 875 | 750 |
| Kubernetes (1,250) / tfidf | 0.734 | 1,144 | 832 | 520 |
| Kubernetes / embed | 0.611 | 832 | 520 | 416 |
| SST-2 (500) / tfidf | 0.682 | 451 | 246 | 328 |

The embedding student's curve is nearly flat from the first batch on AG News and SST-2, so
there's nothing for selection to gain there. For tfidf the diverse variant never did worse
than random and saved 1.1x to 2.3x. Rows selected by the tfidf probe and used to train the embed
student scored within a point of random selection at every budget on CFPB, Kubernetes and
20 Newsgroups, slightly ahead on CFPB and slightly behind early on 20 Newsgroups.

### Other selection strategies

The published benchmarks I read found margin among the safest uncertainty methods on text,
BADGE the only strategy that never falls below random, and coreset, entropy and committee
methods not reliably better than margin. Rerun on these pools the same way (one random
seed, twelve batches, tfidf student), labels needed to reach the full-pool accuracy minus
one point:

| dataset | random | margin | margin+diverse | entropy | BADGE | coreset | committee |
|---|---|---|---|---|---|---|---|
| AG News (5,000) | 2,496 | 1,248 | 1,248 | 1,248 | 1,664 | 2,080 | 832 |
| CFPB (1,500) | 1,375 | 500 | 500 | 1,125 | 1,000 | 625 | 500 |
| Kubernetes (1,250) | 936 | 624 | 624 | 728 | 832 | 936 | 728 |
| 20 Newsgroups (1,250) | 1,144 | 832 | 1,144 | 936 | 1,144 | 1,040 | 936 |
| SST-2 (500) | 287 | 246 | 287 | 246 | 246 | 287 | 369 |

BADGE used k-means++ seeding on the logistic-regression gradient embedding over SVD-256
features of the pool. Coreset was k-center greedy on the same features. The committee
prioritized rows where a tfidf probe and an embedding probe disagreed. Margin wins or ties
everywhere, which matches the published results. The clustering step in the shipped
selector comes out even with plain margin here (tied on three datasets, behind on two,
ahead on kubernetes in the earlier three-seed run). I kept it for pools with near-duplicate
rows, which none of these datasets have.

### Accuracy per cost: which teacher, which rows

The same question in terms of cost, with a cheaper teacher added. Every row of the CFPB base
window with a Sonnet label (2,808) and the 5,000-row AG News pool were relabeled by haiku
4.5, with a prompt that also asks for a 0-1 confidence. Holdouts are the two CFPB drift
windows plus unused base rows (3,254 rows) and the 1,500-row AG News holdout. Per-row
prices are the measured litellm costs.

| | CFPB | AG News |
|---|---|---|
| haiku label accuracy vs gold | 0.785 | 0.837 |
| Sonnet label accuracy vs gold | 0.826 | 0.922 |
| haiku confidence AUROC on its own errors | 0.75 | 0.78 |
| haiku price per row, as a share of Sonnet's | 68% | 68% |

Labeling cost for the tfidf student to reach the Sonnet-labeled full-pool accuracy minus one
point, relative to Sonnet labeling rows in order:

| strategy | CFPB | AG News |
|---|---|---|
| Sonnet, rows in order | 1.00x | 1.00x |
| Sonnet, rows chosen by student margin | 0.33x | 0.50x |
| haiku, rows in order | never (0.766 at full pool) | never (0.828) |
| haiku, rows chosen by student margin | never | never |
| haiku, Sonnet escalation on the least confident 20% | 1.32x | never |
| haiku, Sonnet escalation on the least confident 40% | 1.11x | 1.41x |

Two takeaways. First, cheaper labels didn't pay off. The price gap between haiku and
Sonnet is small on real text, the worse labels cost the student 1.6 to 5 points, and
haiku's confidence isn't good enough to decide what to escalate, so a tiered teacher ends
up costing as much as Sonnet alone. Second, choosing rows did pay off. With the same
teacher, picking rows by the student's uncertainty hit the target for a third of the cost
on CFPB and half on AG News, and peaked above the full-pool student on both (0.785 vs
0.782, 0.881 vs 0.877) at about 60% of the full-pool spend. That's why
`label(budget_usd=...)` changes which rows get labeled, not which teacher labels them.

## Selective prediction

Every student has `predict_proba`, and its top probability ranks rows by how sure it is. Accuracy of the saved students on the same gold holdouts as above, keeping only the
most confident share of rows, with thresholds read off the holdout:

| dataset (rows) | teacher, all rows | student | 100% | 90% | 80% | 70% | 50% |
|---|---|---|---|---|---|---|---|
| CFPB (1,309) | 0.811 | tfidf | 0.759 | 0.793 | 0.810 | 0.835 | 0.858 |
| | | setfit | 0.784 | 0.825 | 0.877 | 0.901 | 0.925 |
| | | encoder | 0.754 | 0.795 | 0.836 | 0.864 | 0.905 |
| | | embed | 0.682 | 0.715 | 0.746 | 0.767 | 0.812 |
| Kubernetes (2,659) | 0.782 | tfidf | 0.733 | 0.766 | 0.794 | 0.828 | 0.884 |
| | | embed | 0.602 | 0.635 | 0.661 | 0.682 | 0.739 |
| 20 Newsgroups (1,200) | 0.885* | setfit | 0.824 | 0.857 | 0.901 | 0.923 | 0.947 |
| | | encoder | 0.807 | 0.848 | 0.878 | 0.909 | 0.950 |
| | | tfidf | 0.782 | 0.826 | 0.862 | 0.890 | 0.942 |
| SST-2 (1,200) | 0.970* | setfit | 0.917 | 0.947 | 0.953 | 0.958 | 0.972 |
| | | tfidf | 0.677 | 0.696 | 0.717 | 0.744 | 0.777 |
| AG News (1,500) | 0.922* | encoder | 0.886 | 0.924 | 0.941 | 0.946 | 0.963 |
| | | tfidf | 0.877 | 0.910 | 0.927 | 0.937 | 0.947 |

\* teacher agreement with gold on the pool rather than on these rows.

On the organic-label tasks, the tfidf student's accuracy on its most confident 80% matches
the teacher's accuracy on all rows. Those are different sets of rows, so it doesn't mean
the student matches the teacher on the rows it keeps. In a separate CFPB calculation,
answering the kept rows locally and sending the rest to the teacher held overall accuracy
around 0.81 at every coverage level I tried, with fewer teacher calls. Before using this
for real, pick the threshold on validation data, then check both models on the same kept
rows, and the whole setup with its fallback on a fresh holdout.

Averaging the probabilities of all four students gained about a point at full coverage on
CFPB, 20 Newsgroups and AG News and lost on SST-2 and Kubernetes, so shrewd ships no
ensemble.

## What the confidence signals are worth

`votes=3` asks the teacher three times and reports how often it agreed with itself. In the
tests against hidden gold, the Claude teachers ran at temperature 1 and still voted almost
unanimously even when wrong, so agreement barely told right labels from wrong ones (AUROC
0.53 on Sonnet 5). The model's own stated 0-1 confidence did much better (AUROC 0.87 to
0.93). Neither one improved the student when used to filter or weight training rows. Even
knowing exactly which teacher labels were wrong gained under a point at these pool sizes,
for tfidf, embed and encoder students alike. The students are short on data more than they
are hurt by noisy labels, so shrewd doesn't use either signal in training.

### Candidate label sets

Another published approach is to ask the teacher for a set of plausible labels when it's
unsure, and train the student on the set (CanDist, arXiv 2506.03857, reports +5 to 6
points with a RoBERTa student at 5k to 11k rows). Sonnet relabeled the 1,500-row CFPB pool
with a primary label, optional alternatives and a confidence. It gave alternatives
on 39% of rows, and the gold label sits inside the candidate set 92.9% of the time against
83.9% for the primary alone. The primary is right 70% of the time on rows with
alternatives and 93% without, so the set is a decent uncertainty flag. Student
accuracy on 5,094 gold rows:

| training labels | tfidf | embed |
|---|---|---|
| primary label | 0.770 | 0.705 |
| one weighted row per candidate | 0.770 | 0.704 |
| refine: student picks within the set, 3 rounds | 0.769 | 0.686 |
| drop rows with alternatives | 0.752 | 0.686 |
| gold on every row (ceiling) | 0.786 | 0.712 |

None of these turned the set into better accuracy, and even perfect labels only add 1.6
points for tfidf. This is the third time better labels failed to help on these pools,
after confidence weighting and the evidence-based relabeling experiment. The students are
short on data, which is why choosing rows is what helps.

## Prompt optimization

`optimize()` runs GEPA against the dev split. To see what it's worth I scored the
seed prompt, the shipped prompt, and prompts from alternative GEPA configurations on 500
gold rows per task (standard error about 1.8 points, and a frontier teacher
sampling at temperature 1 moves the same prompt by up to 0.8 points between runs).

| task, teacher | seed prompt | stock GEPA (400 calls) | best variant | what the variant changed |
|---|---|---|---|---|
| AG News, haiku | 0.782 | 0.842 | 0.852 | Fable 5.1 as reflector |
| Kubernetes, Sonnet 5 | 0.768 | 0.768 | 0.802 | Fable 5.1 as reflector |
| 20 Newsgroups, Sonnet 5 | 0.874 | 0.882 | 0.874 | nothing helped |

Three takeaways. With a cheap teacher the optimizer was worth about 6 points every time.
Minibatch size, validation split, reflection template and sampling strategy all landed
within noise of each other on AG News. With a frontier teacher it's hit or miss. On
kubernetes, where the teacher is the bottleneck, a stronger reflection model turned no
gain into +3.4 points, the same lift as switching the whole teacher to Fable at a quarter
of the labeling cost. On 20 Newsgroups nothing moved gold at all. And the dev-val score
overstates the gain. Newsgroups dev-val rose 5 to 6 points in every configuration while
gold stayed flat, and two kubernetes prompts with the same dev-val differed by 3.6 points
on gold. A 200-row dev split can't rank prompts a few points apart. The locked test set
and the pool-vs-gold check are how you tell whether the optimizer did anything.

Two things explain the earlier runs that produced only one to four candidates. GEPA
accepts a proposal only if it beats its parent on the reflection minibatch, and the
default minibatch is 3 rows: with a teacher that is right 90% of the time, most 3-row
batches are already perfect and the step is skipped. And validating on half the dev split
(the old default) doubled the selection noise. Raising the budget from 300 to 1,200 calls
under that split lowered gold accuracy from 85.5% to 83.5%. The defaults validate on
the whole dev split with 10-row minibatches and an 800-call budget, which evaluates three
or four candidates and no longer overfits at 1,600 calls (0.844 vs 0.842 on gold).

## Calibrated decisions

The second path (`Decisions`) asks a panel of typed questions about each document and
distills the answers into one featurization with a calibrated head per question. The
numbers below are from GoEmotions, 58k Reddit comments each judged by several crowd
raters. That fits this well, since it has many yes/no questions per document and the
ground truth is a rate (4 of 5 raters said anger) rather than a single label.

Eight questions, base rates 2.5% to 25%. Claude Sonnet 5 judged a 1,500-comment pool
(one call per comment, all eight questions). Scored on 4,000 held-out comments with
rater-majority gold.

| | accuracy | ECE | Brier | AUROC | \|p - rater rate\| |
|---|---|---|---|---|---|
| student, raw probabilities | 0.904 | 0.122 | 0.090 | 0.765 | 0.208 |
| **student, calibrated** | **0.931** | **0.043** | **0.058** | 0.765 | **0.114** |
| + teacher also calibrated | 0.937 | 0.038 | 0.056 | 0.725 | 0.090 |

Means over the eight questions. Calibration is fit by cross-fitting on the judged pool,
using the teacher's argmax answer as the target, with human corrections replacing that
target where supplied. It costs no API calls. The scores above are measured against
human annotations on a separate holdout: calibrating to the teacher also improved fit
to those annotations here, but this is not guaranteed when the teacher is systematically
wrong. It cuts ECE by 2.8x and the distance to the human rate by 1.8x. Unlike the
multi-class case it also raises accuracy (0.904 to 0.931), because calibrating a yes/no
answer below 0.5 flips it. That's the right call when a model says 0.55 about something
that happens 3% of the time.

Calibrating the teacher's own probabilities first gets another 2 points closer to the
human rate but costs 4 points of AUROC, because pushing a rare question's targets toward
"no" wipes out the signal the head was learning from. It's off by default.

### The student ends up better calibrated than its teacher

Same 500 comments, teacher and student scored side by side:

| | ECE | Brier | AUROC |
|---|---|---|---|
| Claude Sonnet 5 (verbalized probabilities) | 0.099 | 0.073 | **0.837** |
| tfidf student, calibrated | **0.043** | **0.055** | 0.701 |

The teacher is clearly better at ranking. It knows which comments are angry. It's worse
at stating a probability, because it doesn't know how often it's right. Pooled over all
eight questions, here's what the teacher says against what actually happens:

| teacher says | documents | actually yes |
|---|---|---|
| 0.05 | 500 | 0.048 |
| 0.14 | 500 | 0.064 |
| 0.29 | 500 | 0.112 |
| 0.63 | 500 | 0.268 |

It's roughly 2.3x overconfident whenever it leans yes. Compare that with
[What the confidence signals are worth](#what-the-confidence-signals-are-worth). On
balanced multi-class classification the same model's stated confidence is well calibrated
(ECE 0.030-0.060) and leans under. On rare, subjective yes/no questions it's badly over.
So how well a model's stated confidence holds up depends on the kind of question you ask.

### Which lever moves what

Same pool, same questions, same 4,000-row holdout. Two things were varied: hard argmax targets vs.
soft targets (training on the teacher's full distribution, one weighted row per option),
and tf-idf vs. static embeddings.

| featurization | targets | accuracy | ECE | Brier | AUROC |
|---|---|---|---|---|---|
| tfidf | hard | 0.931 | 0.051 | 0.058 | 0.765 |
| tfidf | soft | 0.929 | 0.054 | 0.059 | 0.786 |
| embed | hard | 0.925 | 0.058 | 0.061 | 0.807 |
| embed | soft | 0.925 | 0.058 | 0.060 | **0.818** |
| Sonnet 5 teacher | | | *0.099* | *0.073* | *0.837* |

All calibrated. ECE stays within 0.051-0.058 and accuracy within 0.925-0.931, while AUROC
moves more. So similar calibration scores can hide different ranking quality. Both the
featurizer and hard vs soft targets change the ranking here. Calibration fixes the
probability scale but doesn't necessarily fix the order of examples.

Soft targets are worth about 2 points of AUROC on tf-idf and 1 on embeddings, and nothing
on the other metrics. That adds to the earlier finding that these students are short on
data more than hurt by noisy labels. The teacher's full distribution does carry more than
its top answer, but it only shows up in ranking.

Per-question AUROC, where the remaining gap to the teacher actually lives:

| question | tfidf | embed | teacher |
|---|---|---|---|
| gratitude | 0.978 | 0.969 | 0.947 |
| amusement | 0.876 | 0.903 | 0.899 |
| anger | 0.756 | 0.869 | 0.873 |
| admiration | 0.824 | 0.834 | 0.860 |
| annoyance | 0.642 | 0.755 | 0.716 |
| curiosity | 0.834 | 0.792 | 0.927 |
| neutral | 0.639 | 0.691 | 0.720 |
| approval | 0.574 | 0.643 | 0.755 |

Embeddings close most of the gap and beat the teacher on gratitude, amusement and
annoyance. What's left is mostly in the two most contested questions (`approval`,
`neutral`) plus `curiosity`, and calibration doesn't help there.

### Cost and latency

500 comments x 8 questions = 4,000 judgments:

| | wall clock | per document |
|---|---|---|
| Sonnet 5, 12-way concurrent | 82.8 s | one API call |
| tfidf student, one CPU core | 0.022 s | 0.044 ms (5.5 us per judgment) |

That's 3,755x faster and free, with the calibration shown above and a lower Brier than
the teacher. The panel also saves on the teacher side. One call answers all eight
questions, so judging the pool cost about an eighth of asking each question separately.

### Stacking a zero-shot second opinion, gated on gold

`distill(zero_shot=True)` adds an NLI model's opinion (`deberta-v3-base-zeroshot-v2.0`) to
each head and keeps it per question only where it beats the plain head on the gold dev
split by 0.02: AUROC for yes/no questions, macro-F1 for multi-class. Both validations ran
through the shipped code path, reusing cached teacher answers (no new API calls).

**GoEmotions, 8 yes/no questions, 4,000-row gold holdout, tf-idf head:**

| | mean AUROC | mean ECE | stacked on |
|---|---|---|---|
| plain head | 0.755 | 0.077 | none |
| every question stacked (experiment) | 0.843 | 0.074 | 8/8 |
| **gated, as shipped** | **0.836** | **0.055** | 5/8 |
| Claude Sonnet 5 teacher | *0.837* | *0.099* | |

| question | dev AUROC gain | gate | holdout AUROC |
|---|---|---|---|
| annoyance | +0.156 | kept | 0.613 -> **0.821** |
| anger | +0.155 | kept | 0.749 -> **0.922** |
| amusement | +0.095 | kept | 0.871 -> 0.969 |
| approval | +0.067 | kept | 0.571 -> 0.646 |
| admiration | +0.050 | kept | 0.804 -> 0.878 |
| curiosity | +0.018 | rejected | 0.834 (stack would be 0.886) |
| gratitude | +0.011 | rejected | 0.978 |
| neutral | +0.000 | rejected | 0.639 |

The gains show up where the head was weakest (the rare, contested questions), and the gate
says no where tf-idf was already doing fine. `curiosity` at +0.018 is the one case where the
0.02 margin clearly cost something. 0.02 was a first guess, and this is the row that would
make me change it.

**Kubernetes issue types, 5-way choice, 1,000-row gold holdout:** dev macro-F1 gain -0.017,
**rejected**. The shipped student is the plain head, with accuracy 0.734 and macro-F1 0.711.
Stacking everything had given 0.708 / 0.668, worse than plain, so the gate got it right.

Why the gate uses gold and not the pool: on Kubernetes, when the teacher disagrees with
the maintainers, the zero-shot model is also wrong 77% of the time. Its mistakes line up
with the teacher's more than the head's do, even though the head was trained on the
teacher's labels. Both are language models reading the text, while the maintainers label
by convention. Pool agreement said the stack was worth +4 points there, and gold said -0.3.
Any metric scored against the teacher will overrate this combination.

Why AUROC for yes/no: with a 2.5% base rate there are ~8 positives in 325 dev rows, and
macro-F1 on the top answer barely moves no matter how good the probabilities get. With an
F1 gate, `anger` was rejected at +0.001 and missed out on +0.17 AUROC.

Cost: 38 ms per document with five stacked heads, against 0.044 ms for the plain student,
since it runs a transformer once per (document, option). It needs the `[encoder]` extra and
is off unless you turn it on.

### Every student on every dataset

Eight datasets through the shipped code with cached teacher answers (no API calls beyond the
first judging), one eval set each (700-4,000 gold rows), all students calibrated. Yes/no and
GoEmotions report AUROC, the rest accuracy. The last column is the open Laya checkpoint
answering the same questions cold, on the same rows.

| dataset | tf-idf | embed | embed+stack | encoder | encoder+stack | Laya zero-shot |
|---|---|---|---|---|---|---|
| SST-2 | 0.766 | 0.826 | 0.973 | 0.897 | **0.976** | 0.496† |
| SMS spam | **0.997** | 0.994 | 0.994 | 0.992 | | 0.969 |
| AG News | 0.870 | 0.857 | 0.857 | **0.889** | | 0.935‡ |
| 20 Newsgroups | 0.801 | 0.764 | 0.781 | **0.814** | | 0.603 |
| CFPB | 0.854 | 0.747 | 0.815 | **0.859** | | 0.645 |
| Kubernetes | 0.717 | 0.634 | 0.696 | **0.764** | | 0.583 |
| GoEmotions | 0.785 | 0.821 | **0.853** | 0.811 | 0.848 | 0.874 |
| SST-5 (MAE, lower is better) | 0.978 | 0.871 | | **0.657** | | 1.000 |

† Laya's yes/no path returns 0.000 for a question phrased as a question. Posed three ways
per question and credited with the best (generous to Laya, since the pick is made on the
eval rows), it scores SST-2 0.947, SMS 0.974, GoEmotions 0.907 (ECE 0.22 /
0.09 / 0.20). ‡ AG News is in Laya's training mix, by its own model card.

ECE on the same grid: every student here lands between 0.002 and 0.064, and Laya between
0.05 and 0.56. (Its card says to refit a temperature on your own data first, which is what
the calibration step here does.) Latency per document: tf-idf 0.04-0.5 ms, embed
0.01-0.08 ms, encoder 3-29 ms, the NLI stack 13-812 ms (the 20-class sets are the slow
ones), Laya 76-276 ms.

The students were trained and calibrated on each task, while the Laya checkpoint answered
zero-shot. Training data, architecture, and calibration all differ, so this comparison
cannot tell us which architecture is better on its own. To compare calibration methods,
both sides would need comparable calibration data. These numbers describe the checkpoint
used in these experiments.

What I take from the grid:

- **The encoder is the best student wherever word counts leave room to improve.** AG News
  +1.9, Newsgroups +1.3, CFPB +0.5, Kubernetes +4.7 over tf-idf, and SST-5 a third of a
  level. It's the only student that beat tf-idf on CFPB and Kubernetes at all.
- **The NLI stack only helps where the base is weak and the NLI model is strong,** which
  here means sentiment and emotion. On SST-2 it takes both featurizers to 0.97
  (encoder+stack 0.976, embed+stack 0.973), so the NLI model is doing the work. On
  GoEmotions embed+stack (0.853) still beats encoder+stack (0.848) at a quarter of the
  latency. On every multi-class set the gate kept it and it still lost to plain tf-idf,
  which is why I didn't run it over the encoder on the other five.
- **Against the Laya checkpoint I tested,** the task-trained students scored better on six
  of eight datasets (SST-2 0.973 vs 0.947, SMS 0.997 vs 0.974, and the four domain sets by
  13-21 points), had lower ECE on all eight, and were 100-25,000x faster when `auto` picked
  tf-idf or embeddings. The scores mix AUROC, accuracy and MAE as labeled above. Laya wins
  AG News (which is in its training data) and clearly wins GoEmotions, 0.907 against 0.858
  here with Fable labels and the stack. Emotion in short comments is where a broadly
  trained 395M cross-encoder beats a panel distilled from 1,500 documents. The stack
  accepts any scorer, so gating Laya's scores onto a GoEmotions head is the next thing
  I'd try.

### The fine-tuned encoder, and GEPA on the panel

`features="encoder"` fine-tunes ModernBERT-base on the 1,825 training documents (3 epochs
in this experiment, the default is 6) with log + spherical + ranked-probability loss on the
teacher's soft distributions, then fits linear heads on the pooled representation,
cross-fit calibrated with three folds. Same 4,000-row gold holdout:

| student | mean AUROC | mean ECE | ms / document |
|---|---|---|---|
| tf-idf | 0.755 | 0.077 | 0.04 |
| static embeddings | 0.807 | 0.058 | 0.05 |
| **fine-tuned encoder** | **0.806** | **0.048** | 2.9 |
| tf-idf + gated zero-shot stack | 0.836 | 0.055 | 38 |
| Sonnet 5 teacher | *0.837* | *0.099* | *~150-300* |

On this panel fine-tuning improved calibration but not ranking. It ties static embeddings
on AUROC, has the best ECE of any student, and takes 65x as long as tf-idf while still
running 25-100x faster than the hosted model. It clearly wins on `curiosity` (0.924 vs
0.834) and loses on `neutral`. The contested questions (`approval` 0.611, `annoyance`
0.742) stay hard for every student. Short Reddit comments and 1,825 rows may just not give
a 150M encoder much to learn beyond what static embeddings already have. The zero-shot
stack still helps more.

`optimize()`: GEPA over the prompt header with Fable 5.1 reflecting, budget 800 calls,
objective mean of 1 - Brier/2 on the 325-row dev split. Dev score went from 0.926 to 0.935, three
candidates. The gain is small because easy "no"s dominate the score, but the header
it wrote is interesting. From dev mistakes alone it concluded that "admiration is
systematically over-predicted ... keep it at or below ~0.5", "approval is also over-predicted
... default 0.3-0.4", and "rare emotions with no textual cue belong at 0.02-0.05". In other
words, it found the teacher's 2.3x overconfidence on rare questions by itself and wrote
instructions to correct it.

Measured afterwards on 575 gold rows the optimizer never saw, and on the 4,000-row holdout
after re-judging the pool under each header and distilling the tf-idf student:

| header | words | teacher AUROC | teacher Brier | student AUROC | student ECE | student Brier | cost/doc vs default, no cache | cost/doc vs default, cached |
|---|---|---|---|---|---|---|---|---|
| default | 194 | 0.838 | 0.0742 | 0.785 | 0.038 | 0.0550 | 1.00x | 1.00x |
| GEPA, uncapped | 767 | **0.864** | **0.0653** | **0.793** | **0.029** | **0.0522** | 2.39x | 1.84x |
| GEPA, `max_header_words=250` | 248 | 0.849 | 0.0699 | n/a | n/a | n/a | 1.15x | 0.74x |

It didn't overfit. The teacher's held-out AUROC rose on 7 of 8 questions and Brier fell
12%. The student kept about a third of the ranking gain (its tf-idf features hold it back)
and all of the calibration gain.

The cost columns are Sonnet 5 judging GoEmotions comments. Before the system prompt was
marked for the provider's cache, the 767-word header cost 2.4x the default per document,
and since the header goes out with every document for the life of the project, that
looked like a real problem. With the prefix cached (2,508 of 2,626 prompt tokens read from
cache at 10% of the input price) the premium drops to 1.8x, and what's left isn't input.
The teacher writes 276 output tokens per document under the long header against 152 under
the default, for a JSON answer of the same length. The extra is the model's own reasoning,
billed as output. The header has no cap by default because the uncapped one did best on
gold. Use `max_header_words` when the per-document cost matters more than the last 0.015
of teacher AUROC. The 248-word header was even cheaper per document than the default.

### Teacher choice and the effect of tied probabilities

Same GoEmotions panel, same 1,500 pool documents, same prompt, Fable 5.1 as the teacher
instead of Sonnet 5, at 2.2x Sonnet's cost to judge the pool with the system prompt cached:

| | teacher AUROC (575 gold) | teacher Brier | tf-idf student | embed student | embed+stack student |
|---|---|---|---|---|---|
| Sonnet 5 | 0.838 | 0.0742 | 0.785 / ECE 0.038 | 0.821 / 0.044 | 0.853 / 0.051 |
| **Fable 5.1** | **0.868** | **0.0619** | **0.803 / 0.026** | **0.831 / 0.026** | **0.858 / 0.030** |

The student kept part of the teacher's ranking gain and improved on Brier and ECE. Here,
switching teachers moved the student about as much as prompt optimization did. I haven't
tested whether the two add up.

A teacher that writes out its probabilities only uses about 20 distinct values per
question (70% of `gratitude` answers are exactly 0.02), and AUROC counts every tied yes/no
pair as half correct. Breaking Sonnet 5's ties using the true labels would lift it from
0.838 to 0.870 on the same rows. Fable 5.1 uses ~50% more distinct values and its tie gap
is half as large. That calculation uses the right answers to put tied examples in the best
possible order, and I have no evidence the teacher could find that order itself. Its
measured AUROC is still 0.838, and that's the number to use when comparing it with a model
that gives continuous scores.

### The four pre-built panels

Each panel has 500 labeled seed rows (325 dev / 175 test), a 1,500-document unlabeled pool
judged by Fable 5.1, and a separate holdout of up to 4,000 reference-labeled rows kept out
of training and prompt optimization. I ran `auto` featurization plus the gated stack, then
an encoder variant with no new API calls (the teacher's answers are cached). `pick.py` chose between them
on the seed test split, with ties within 0.005 going to the faster model, which makes that
split a validation set. The table shows the separate holdout scores. For messages I
overrode the pick by hand after looking at scam texts, as noted in its row.

| panel | source (Hub) | gold | tf-idf / embed | encoder | shipped |
|---|---|---|---|---|---|
| messages | ucirvine/sms_spam | unsolicited | 0.991 AUROC, ECE 0.004, 0.04 ms | **0.997, 0.002, 6.3 ms** | encoder (tied on gold, but on the README's parcel scam tf-idf says 0.64 "personal" and the encoder 0.94 "scam") |
| email | zefang-liu/phishing-email-dataset | spam | **0.993**, 0.010, 0.5 ms | not run (long emails, 1 h per fit) | tf-idf |
| guardrail | lmsys/toxic-chat + safe-guard-prompt-injection + jackhhao/jailbreak | harmful / jailbreak / injection | **0.937 / 0.971 / 0.987**, 0.03 ms | 0.941 / 0.981 / 0.996, 72 ms | embed (won on gold test) |
| pii | ai4privacy/pii-masking-200k | contact / identity / financial / credentials / device | 0.864 / 0.739 / 0.928 / 0.939 / 0.920, 0.09 ms | **0.941 / 0.921 / 0.963 / 0.965 / 0.995**, 9.4 ms | encoder |

The panels showed three things the benchmark datasets hadn't:

- The zero-shot stack was rejected on all twelve gold questions. Over a decent featurizer
  it added nothing here, so the encoder run skips the stacked version.
- A dataset's label isn't always the question you wrote. The email corpus's "Phishing
  Email" class is spam in the broad sense (casino bonuses, snoring cures, address lists).
  Asked the stricter question, the teacher put every safe email under 0.1 and half the
  "phishing" ones under 0.5. So I moved the gold to a `spam` question, and `phishing`
  itself only has teacher labels.
- The pii panel is the clearest case for the encoder. `identity` scores 0.739 with tf-idf,
  0.740 with character n-grams and 0.921 fine-tuned. Spotting names and birth dates takes
  context that character patterns miss.

### What does not work

- **`votes=3` on a decision panel.** It was already a weak signal for classification
  (AUROC 0.53), and I had no reason to expect better here. The ensemble backend averages
  distributions instead, which keeps the disagreement that counting votes throws away.
- **Calibrating on the seed dev split.** On the classification holdouts it made two of
  three datasets worse than doing nothing, because 78-195 unusually easy rows fit a scale
  that real data doesn't match. `Decisions` calibrates on the
  judged pool for the same reason.
- **A low ECE on its own.** Two of the eight questions (`approval`, `annoyance`) end
  up near-perfectly calibrated and near-useless, AUROC 0.505 and 0.469. The head
  answers the base rate on every comment. They're the two most contested questions in
  the set, where the crowd raters themselves disagree 17-23% of the time. `distill()`
  reports this as a `fail` finding rather than as a good ECE.

## Compared to other tools

shrewd does four things: picks pool rows for the LLM to label under a dollar budget,
distills to a small local classifier scored against gold labels, lets the student abstain
based on its confidence, and optimizes the prompt inside the labeling loop. Here's how the
tools I looked at compare:

- [TRACER](https://github.com/adrida/tracer) learns a surrogate from production LLM
  classification traces and defers uncertain inputs to the LLM behind a calibrated gate.
  It evaluates parity with the teacher, not against gold labels, has no labeling budget,
  and does not optimize prompts. Choose it when you already run an LLM classifier and want
  the LLM fallback kept. I ran both on eight datasets. Where both shipped a model on short
  text the results were close. On long organic text its gate declined to deploy, while
  shrewd's tfidf student scored 0.718.
- [DSPy](https://dspy.ai/tutorials/classification_finetuning/) `BootstrapFinetune` with
  GEPA covers teacher labeling, prompt optimization and distillation into a small LLM
  student rather than a CPU classifier, with no budgeted acquisition or abstention.
- [Autolabel](https://github.com/refuel-ai/autolabel) estimates cost and reports the
  LLM's own confidence. [distilabel](https://github.com/argilla-io/distilabel)
  has a text-classification labeling task with no evaluation. Use it for synthetic-data
  pipelines. [Cleanlab TLM](https://help.cleanlab.ai/tlm/) scores the LLM's outputs and
  plots accuracy against the share auto-labeled. The abstention is on the LLM side, and
  there is no small model.
- [Prodigy](https://prodi.gy/docs/large-language-models), Label Studio and Snorkel do
  uncertainty-ordered annotation with a human oracle, LLM pre-annotation, and (Snorkel,
  Prodigy) a small model trained on the result. None spends an LLM budget by student
  uncertainty.
- The closest workflow I found to budgeted acquisition was FutureSearch's
  [writeup and SDK](https://futuresearch.ai/active-learning-llm-oracle/): entropy sampling
  to an LLM oracle with a per-label price and no cap. The idea also appears in a paper
  ([arXiv 2511.11574](https://arxiv.org/abs/2511.11574)) without code.
- [SetFit](https://github.com/huggingface/setfit) on the seed alone and
  [GLiClass](https://github.com/Knowledgator/GLiClass) zero-shot are the "skip the LLM"
  options. The seed-only baselines above measure the first: 3 to 24 points behind on CFPB.

I wanted budgeted labeling, local students, evaluation against human labels, abstention,
and prompt optimization in one place, so shrewd puts them together. Other tools cover
parts of this and can be extended. This list only covers the ones I tried or read about.
