"""Smoke-test the dashboard's aggregate views and local text scoring interaction."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def main() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    if app.exception:
        raise RuntimeError(app.exception[0].message)
    assert any("CivilComments under subpopulation shift" in title.value for title in app.title)

    text_inputs = [widget for widget in app.text_area if widget.key == "local_text_input"]
    assert len(text_inputs) == 1
    text_inputs[0].set_value("A short user supplied sentence for a local smoke check.")
    app.button[0].click().run()
    if app.exception:
        raise RuntimeError(app.exception[0].message)
    assert any("ERM score" in metric.label for metric in app.metric)
    assert any("Group-reweighted score" in metric.label for metric in app.metric)
    assert any("Explain a model score" in widget.label for widget in app.selectbox)
    print("Dashboard AppTest passed: load, local scoring, and explanation selector.")


if __name__ == "__main__":
    main()
