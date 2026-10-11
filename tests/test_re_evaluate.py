"""Offline evaluation regression tests, without model downloads."""
import json
import pandas as pd
import pytest
from mami.multilabel.re_evaluate import re_evaluate


def test_re_evaluate_fenced_responses(tmp_path):
    source = tmp_path / "raw.csv"
    payload = '{"shaming":1,"stereotype":0,"objectification":0,"violence":0}'
    pd.DataFrame([{
        "response": "```json\n" + payload + "\n```",
        "true_shaming": 1, "true_stereotype": 0,
        "true_objectification": 0, "true_violence": 0,
    }]).to_csv(source, index=False)
    result, destination = re_evaluate(source)
    assert result["valid_responses"] == 1
    assert result["exact_match_accuracy_all"] == 1.0
    assert json.loads(destination.read_text()) == result
    with pytest.raises(FileExistsError):
        re_evaluate(source)


def test_re_evaluate_rejects_missing_columns(tmp_path):
    source = tmp_path / "incomplete.csv"
    pd.DataFrame([{"response": "{}"}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Missing saved prediction columns"):
        re_evaluate(source)
