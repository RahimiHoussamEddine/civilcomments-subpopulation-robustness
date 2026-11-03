"""Local-only dashboard for aggregate robustness results and user-entered text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from civilcomments_robustness.inference import feature_contributions, probability_for

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/civilcomments_models.joblib"
VALIDATION_PATH = ROOT / "reports/validation_metrics.json"
EVALUATION_PATH = ROOT / "reports/evaluation.json"

st.set_page_config(
    page_title="CivilComments Robustness Workbench",
    page_icon="🧭",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the local sparse text models…")
def load_models() -> dict[str, Any]:
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ci_text(metrics: dict[str, Any], metric_name: str) -> str:
    interval = metrics[metric_name]
    return (
        f"95% article-cluster interval {interval['ci_95_lower']:.3f}–{interval['ci_95_upper']:.3f}"
    )


if not all(path.exists() for path in (MODEL_PATH, VALIDATION_PATH, EVALUATION_PATH)):
    st.title("CivilComments under subpopulation shift")
    st.info(
        "The local model and aggregate evaluation reports are not ready yet. "
        "From the project directory, run `make fetch`, `make validate-data`, "
        "`make train`, then `make evaluate`."
    )
    st.stop()

bundle = load_models()
validation = load_report(VALIDATION_PATH)
evaluation = load_report(EVALUATION_PATH)

st.title("CivilComments under subpopulation shift")
st.caption(
    "A local workbench for comparing empirical risk minimization with group-aware "
    "sample weighting on overlapping identity-mention subgroups."
)
st.warning(
    "Research demonstration only. Identity fields mark mentions in text, not author "
    "demographics. Scores are not moderation decisions, and user-entered text stays "
    "in this local app process."
)

models = evaluation["models"]
erm = models["erm"]["at_validation_selected_threshold"]
reweighted = models["group_reweighted"]["at_validation_selected_threshold"]
erm_ci = models["erm"]["article_cluster_bootstrap_95ci"]
reweighted_ci = models["group_reweighted"]["article_cluster_bootstrap_95ci"]
paired = evaluation["paired_group_reweighted_minus_erm_article_cluster_bootstrap_95ci"]

score_tab, subgroup_tab, validation_tab, local_tab = st.tabs(
    ["Test-set summary", "Subgroups", "Validation choices", "Score local text"]
)

with score_tab:
    st.subheader("Locked test-set comparison")
    st.caption(
        f"{evaluation['test_rows']:,} leakage-screened comments from "
        f"{evaluation['test_articles']:,} source articles. Model and threshold "
        "choices were made with validation data."
    )
    columns = st.columns(4)
    columns[0].metric("ERM accuracy", f"{erm['accuracy']:.3f}")
    columns[1].metric("Group-reweighted accuracy", f"{reweighted['accuracy']:.3f}")
    columns[2].metric("ERM worst-group accuracy", f"{erm['worst_group_accuracy']:.3f}")
    columns[3].metric(
        "Reweighted worst-group accuracy", f"{reweighted['worst_group_accuracy']:.3f}"
    )
    st.markdown(
        f"**Paired average-accuracy difference:** "
        f"{paired['candidate_minus_baseline_average_accuracy']['delta']:+.3f} "
        f"({ci_text(paired, 'candidate_minus_baseline_average_accuracy')})"
    )
    st.markdown(
        f"**Paired worst-group difference:** "
        f"{paired['candidate_minus_baseline_worst_group_accuracy']['delta']:+.3f} "
        f"({ci_text(paired, 'candidate_minus_baseline_worst_group_accuracy')})"
    )

    comparison = pd.DataFrame(
        [
            {"Model": "ERM", **erm},
            {"Model": "Group-reweighted", **reweighted},
        ]
    )
    available = [
        column
        for column in (
            "roc_auc",
            "average_precision",
            "brier_score",
            "log_loss",
            "expected_calibration_error_15_bins",
            "true_positive_rate",
            "true_negative_rate",
            "false_positive_rate",
        )
        if column in comparison
    ]
    st.dataframe(
        comparison[["Model", "threshold", "accuracy", "worst_group_accuracy", *available]],
        hide_index=True,
        width="stretch",
    )
    with st.expander("How to read these results"):
        st.write(evaluation["model_comparison_note"])
        st.write(evaluation["interpretation_note"])
        st.write(
            "Intervals resample source articles as clusters so comments from the "
            "same article are not treated as independent observations."
        )

with subgroup_tab:
    st.subheader("Performance across overlapping identity-mention groups")
    erm_groups = erm["subgroups"]
    reweighted_groups = reweighted["subgroups"]
    group_rows = []
    reweighted_by_name = {row["group"]: row for row in reweighted_groups}
    for row in erm_groups:
        other = reweighted_by_name[row["group"]]
        identity, label = row["group"].rsplit("|", maxsplit=1)
        group_rows.append(
            {
                "Identity mention": identity,
                "Toxicity label": "Toxic" if label == "1" else "Non-toxic",
                "Group": row["group"],
                "Comments": row["count"],
                "ERM accuracy": row["accuracy"],
                "Reweighted accuracy": other["accuracy"],
                "Difference": other["accuracy"] - row["accuracy"],
            }
        )
    groups_frame = pd.DataFrame(group_rows)
    identities = ["All identity mentions", *sorted(groups_frame["Identity mention"].unique())]
    selected_identity = st.selectbox("Identity mention", identities)
    selected_label = st.selectbox("Toxicity label", ["Both", "Non-toxic", "Toxic"])
    filtered = groups_frame
    if selected_identity != "All identity mentions":
        filtered = filtered[filtered["Identity mention"] == selected_identity]
    if selected_label != "Both":
        filtered = filtered[filtered["Toxicity label"] == selected_label]
    filtered = filtered.sort_values("ERM accuracy", ascending=True)
    st.plotly_chart(
        px.bar(
            filtered,
            x="Group",
            y=["ERM accuracy", "Reweighted accuracy"],
            barmode="group",
            range_y=[0, 1],
            title="Test accuracy at each model's validation-selected threshold",
            labels={"value": "Accuracy", "variable": "Model"},
        ),
        width="stretch",
    )
    st.dataframe(filtered, hide_index=True, width="stretch")
    st.caption(
        "Each identity×label group overlaps other groups. Group accuracy is recall "
        "for toxic comments and specificity for non-toxic comments."
    )

with validation_tab:
    st.subheader("Validation-only model and threshold selection")
    st.caption(
        "The test split was not used to choose the regularization strength, group "
        "weight strength, or global classification thresholds."
    )
    metric_choice = st.selectbox(
        "Validation curve metric",
        [
            "accuracy",
            "worst_group_accuracy",
            "macro_group_accuracy",
            "true_positive_rate",
            "false_positive_rate",
        ],
    )
    curve_rows = []
    for model_name, label in (("erm", "ERM"), ("group_reweighted", "Group-reweighted")):
        for point in validation["threshold_curves"][model_name]:
            curve_rows.append(
                {
                    "Threshold": point["threshold"],
                    "Metric value": point[metric_choice],
                    "Model": label,
                }
            )
    curve_frame = pd.DataFrame(curve_rows)
    st.plotly_chart(
        px.line(
            curve_frame,
            x="Threshold",
            y="Metric value",
            color="Model",
            range_y=[0, 1],
            markers=True,
            title=f"Validation {metric_choice.replace('_', ' ')} by global threshold",
        ),
        width="stretch",
    )
    selected = validation["selected"]
    st.json(
        {
            "ERM": selected["erm"],
            "Group-reweighted": selected["group_reweighted"],
            "Group weight strength candidates": [
                {
                    "strength": candidate["group_weight_strength"],
                    "validation_worst_group_accuracy": candidate["validation_metrics"][
                        "worst_group_accuracy"
                    ],
                }
                for candidate in validation["group_weight_strength_candidates"]
            ],
        }
    )

with local_tab:
    st.subheader("Compare local predictions on text you enter")
    st.caption(
        "Only this supplied text is scored here; public benchmark comments and "
        "row-level predictions are not displayed or stored."
    )
    with st.form("local_text_form"):
        query = st.text_area(
            "Text to score",
            key="local_text_input",
            height=140,
            max_chars=5000,
            placeholder="Enter a short text sample for local model comparison…",
        )
        submit = st.form_submit_button("Score locally")
    if submit:
        if not query.strip():
            st.info("Enter some text first.")
        else:
            results = []
            model_configs = (
                ("ERM", "erm_model", "erm_threshold"),
                (
                    "Group-reweighted",
                    "group_reweighted_model",
                    "group_reweighted_threshold",
                ),
            )
            result_columns = st.columns(2)
            for column, (label, model_name, threshold_name) in zip(
                result_columns, model_configs, strict=True
            ):
                probability = probability_for(bundle, model_name, query)
                threshold = float(bundle[threshold_name])
                above = probability >= threshold
                results.append(
                    {
                        "Model": label,
                        "Estimated toxicity score": probability,
                        "Validation threshold": threshold,
                        "Above threshold": above,
                    }
                )
                column.metric(f"{label} score", f"{probability:.3f}")
                column.caption(
                    f"Validation threshold: {threshold:.3f} · "
                    f"{'above' if above else 'below'} threshold"
                )
            st.dataframe(pd.DataFrame(results), hide_index=True, width="stretch")
            explanation_model = st.selectbox(
                "Explain a model score",
                [("ERM", "erm_model"), ("Group-reweighted", "group_reweighted_model")],
                format_func=lambda item: item[0],
                key="explanation_model",
            )
            contributions = feature_contributions(bundle, explanation_model[1], query)
            if contributions.empty:
                st.caption("No fitted vocabulary features matched this text.")
            else:
                st.dataframe(contributions, hide_index=True, width="stretch")
            st.caption(
                "Feature contributions show terms associated with the model's linear "
                "score. They are not causal explanations or a complete account of a prediction."
            )
