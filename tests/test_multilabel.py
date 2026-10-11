"""Task B parsing and metric smoke tests; no model download required."""
import json
import pandas as pd
from mami.multilabel.parsing import parse_response
from mami.multilabel.evaluation import evaluate


def test_multilabel_parser():
    assert parse_response('{"shaming":1,"stereotype":0,"objectification":1,"violence":0}') == {
        "shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0}
    assert parse_response('{"shaming":true,"stereotype":0,"objectification":1,"violence":0}') is None
    assert parse_response('not json') is None
    assert parse_response('{"shaming":0}') is None


def test_multilabel_metrics_and_invalid():
    truth = pd.DataFrame([{"shaming":1,"stereotype":0,"objectification":0,"violence":0},
                          {"shaming":0,"stereotype":0,"objectification":0,"violence":0}])
    report = evaluate(truth, [{"shaming":1,"stereotype":0,"objectification":0,"violence":0}, None])
    assert report["n"] == 2
    assert report["invalid_responses"] == 1
    assert report["exact_match_accuracy_all"] == 0.5
    assert report["hamming_loss_all"] == 0.5
