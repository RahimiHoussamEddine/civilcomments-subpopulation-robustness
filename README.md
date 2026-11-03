# CivilComments under subpopulation shift

An end-to-end NLP workbench that asks whether group-aware sample weighting can
improve the worst-group behavior of a simple, efficient toxicity classifier.
It compares sparse word TF-IDF with logistic-loss SGD under empirical risk
minimization (ERM) and overlapping identity-mention × label reweighting.

The project uses the official WILDS CivilComments v1.0 benchmark. The dataset
contains public online comments, subjective crowd toxicity annotations, eight
identity-mention fields, and article-disjoint train/validation/test splits. An
identity mention is not an author's demographic label. See [NOTICE.md](NOTICE.md)
for data handling and use limits. The official WILDS page identifies the data as
[CC0 / public domain](https://wilds.stanford.edu/datasets/#civilcomments).

## What this project evaluates

- A word unigram/bigram TF-IDF baseline with L2-regularized `SGDClassifier`
  logistic loss.
- A group-aware variant trained with bounded inverse-frequency weights over
  16 overlapping identity-mention × toxicity-label groups.
- Validation-only selection of regularization, group-weight strength, and a
  global threshold for each model.
- Accuracy, worst and macro group accuracy, ROC-AUC, average precision, Brier
  score, log loss, calibration error, TPR, TNR, and FPR.
- Paired and unpaired percentile intervals that resample source articles as
  clusters, preserving within-article dependence.

## Measured test results

One bounded CPU run selected `alpha=1e-5` for both models, group-weight strength
`2.0`, and threshold `0.36` for each model (the threshold objectives differ as
described below). On 133,781 leakage-screened test comments from 20,943 source
articles, the group-reweighted model traded 3.1 percentage points of average
accuracy for a 35.6 point increase in worst-group accuracy. The 95% paired
article-cluster intervals were [-3.29, -2.92] points for average accuracy and
[+31.81, +37.50] points for worst-group accuracy.

| Model | Accuracy (95% article CI) | Worst-group accuracy (95% article CI) | Macro group accuracy | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| ERM | 0.912 (0.909–0.914) | 0.274 (0.248–0.303) | 0.669 | 0.903 | 0.617 |
| Group-reweighted | 0.881 (0.877–0.884) | 0.629 (0.601–0.640) | 0.746 | 0.892 | 0.594 |

The group-weighted approach also had a higher Brier score (0.076 vs 0.069) and
15-bin expected calibration error (0.089 vs 0.045), and a higher true-positive
rate (0.655 vs 0.362) with lower true-negative rate (0.910 vs 0.982). Results
describe this benchmark and these overlapping text groups; they do not prove
fairness for people or imply a deployment recommendation. Full aggregate values
and intervals are in [`reports/benchmark_results.md`](reports/benchmark_results.md)
and [`reports/evaluation.json`](reports/evaluation.json).

The official WILDS benchmark paper defines the source article split and
subpopulation evaluation protocol: [WILDS paper](https://arxiv.org/abs/2012.07421).
This implementation additionally excludes empty text and exact normalized text
duplicates across splits. It uses the validation and test comment text only to
apply that predeclared duplicate filter; test labels do not influence training,
threshold selection, or model choice. The excluded row counts are written to
`reports/data_quality.json` and the evaluation reports.

## Quick start

Use Python 3.12. Create an isolated environment, install the pinned dependencies,
download the public dataset, and run the leakage and schema checks:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
make fetch
make validate-data
```

`make fetch` downloads the official WILDS CodaLab archive, verifies its pinned
SHA-256 and CSV hash, extracts only the expected CSV, and records aggregate
quality checks. The archive and extracted comments remain under ignored
`data/raw/`; they are never committed.

Train from the official train split, choose models and thresholds using only
validation, then score the test split once:

```bash
make train
make evaluate
```

Training and evaluation artifacts are stored locally under ignored `models/`
and `reports/` paths. The public repository contains code and a source manifest,
not raw comments, row-level predictions, or model binaries.

Launch the interactive dashboard at **http://127.0.0.1:8501**:

```bash
make app
```

The dashboard compares aggregate test results, explores overlapping subgroup
accuracy, plots validation threshold curves, and scores user-entered text locally
with both models. It does not show benchmark comment examples or save the
provided text. Feature contributions are model associations, not causal
explanations.

## Development checks

```bash
make test
make quality
make smoke-app
```

`make smoke-app` needs the local model and reports created by `make train` and
`make evaluate`. CI runs unit tests, dependency checks, and Ruff; it intentionally
does not download the sensitive-text dataset or train models.

## Method and interpretation

The baseline chooses its regularization strength by validation log loss and its
global threshold by validation accuracy. The group-reweighted method uses the
same train-fitted TF-IDF vocabulary and selected regularization strength; the
sample-weight strength and global threshold are selected by validation
worst-group accuracy. The final locked test split is evaluated once after those
choices. See `reports/validation_metrics.json` and `reports/evaluation.json` for
the actual run outputs after training.

Group accuracy is recall for toxic comments and specificity for non-toxic
comments in each identity-mention × label cell. These groups overlap, and the
analysis does not establish fairness for people, causal effects, or suitability
for content moderation. Annotation subjectivity, historical sampling, possible
pretraining overlap in the published benchmark, and the extra normalized-text
duplicate exclusion limit interpretation.

## Reproducibility

- Data and split provenance: `data/source_manifest.json`
- Integrity and leakage report: `reports/data_quality.json`
- Validation choices: `reports/validation_metrics.json`
- Locked test evaluation: `reports/evaluation.json`
- Data and use limits: `NOTICE.md`
- CI: `.github/workflows/ci.yml`
- Measured aggregate outcomes: `reports/benchmark_results.md`
- Local-only Streamlit configuration with usage statistics disabled:
  `.streamlit/config.toml`

All code is MIT-licensed; dataset terms are separate. The upstream WILDS
CivilComments data page lists the benchmark data as CC0.
