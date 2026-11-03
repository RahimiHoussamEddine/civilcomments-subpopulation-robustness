# CivilComments benchmark results

This report contains aggregate metrics only. No comment text, row-level
prediction, or model binary is included.

## Protocol

- Official WILDS CivilComments v1.0 article-disjoint splits.
- Source rows: train 269,038; validation 45,180; test 133,782. After removing
  empty text and cross-split case-folded, whitespace-normalized duplicate text:
  train 267,669; validation 44,992; test 133,781.
- Eight identity-mention fields crossed with binary toxicity labels define 16
  overlapping groups. Identity annotations are not author demographics.
- Word unigram/bigram TF-IDF, 80,000 fitted features, and an L2 logistic-loss
  SGD classifier. Both models use the same vocabulary and `alpha=1e-5`.
- ERM threshold 0.36 maximized validation average accuracy. Group-reweighted
  threshold 0.36 maximized validation worst-group accuracy; sample-weight
  strength 2.0 was selected from `{0.5, 1.0, 2.0}` on validation.
- The locked test was scored after model/threshold selection. It has 133,781
  comments from 20,943 articles. Percentile confidence intervals use 500
  source-article cluster resamples. The paired interval uses the same article
  resample for both models.

## Test metrics at each model's validation-selected threshold

| Metric | ERM | Group-reweighted |
| --- | ---: | ---: |
| Accuracy | 0.9118 | 0.8807 |
| Accuracy, 95% article-cluster CI | 0.9092–0.9142 | 0.8772–0.8842 |
| Worst-group accuracy | 0.2738 | 0.6294 |
| Worst-group accuracy, 95% article-cluster CI | 0.2475–0.3030 | 0.6011–0.6395 |
| Macro group accuracy | 0.6686 | 0.7456 |
| ROC-AUC | 0.9033 | 0.8921 |
| Average precision | 0.6172 | 0.5943 |
| Brier score | 0.0687 | 0.0760 |
| Log loss | 0.2368 | 0.2728 |
| Expected calibration error, 15 bins | 0.0449 | 0.0886 |
| True-positive rate | 0.3621 | 0.6551 |
| True-negative rate | 0.9824 | 0.9097 |
| False-positive rate | 0.0176 | 0.0903 |

## Paired difference (group-reweighted minus ERM)

| Metric | Difference | 95% article-cluster CI |
| --- | ---: | ---: |
| Average accuracy | -0.0311 | -0.0329 to -0.0292 |
| Worst-group accuracy | +0.3556 | +0.3181 to +0.3750 |

## Interpretation

On this benchmark run, reweighting substantially improved the lowest measured
identity-mention × label group accuracy while reducing aggregate accuracy and
probability calibration. This is a measurable tradeoff, not evidence of
fairness for people or an endorsement for content moderation. Groups overlap,
annotations are subjective, and identity fields represent textual mentions.
Exact duplicate exclusions change the official split row counts; their counts
are available in `reports/data_quality.json` and the JSON evaluation report.
