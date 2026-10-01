"""
Tests for dashboard/app.py, the Dense Evolution Dashboard (Streamlit): the app
starts, every section opens, and every button in every section runs, each on
the real engine, with no exception.
"""
import pathlib

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
TIMEOUT = 600


def _app():
    return st_testing.AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def _sections():
    at = _app()
    return list(at.sidebar.radio[0].options)


def _open(section):
    at = _app()
    at.sidebar.radio[0].set_value(section).run()
    return at


def test_app_starts_without_exceptions():
    assert not _app().exception


@pytest.mark.parametrize("section", ["Build", "Results", "Chemistry", "Noise", "System"])
def test_every_section_opens_and_every_button_runs(section):
    at = _open(section)
    assert not at.exception, [str(e.value) for e in at.exception]
    labels = [b.label for b in at.button]
    assert labels, f"no buttons in {section}"
    for label in labels:
        at = _open(section)
        matches = [b for b in at.button if b.label == label]
        if not matches:
            continue
        matches[0].click().run()
        assert not at.exception, (label, [str(e.value) for e in at.exception])


def test_sections_list_is_the_expected_one():
    assert _sections() == ["Build", "Results", "Chemistry", "Noise", "System"]
