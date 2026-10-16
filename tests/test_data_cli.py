"""CLI boundary checks for saved memberships and JSON rankings."""

import json
from importlib.resources import files

import pytest

from atlasir.cli import main


def args(command):
    return [
        command,
        "--data",
        str(files("atlasir").joinpath("fixtures", "beir")),
        "--name",
        "fixture",
        "--role",
        "train",
    ]


def test_cli_selection_is_reusable_and_refuses_overwrite(tmp_path, capsys):
    destination = tmp_path / "splits" / "source.json"
    assert main([*args("make-splits"), "--output", str(destination)]) == 0
    capsys.readouterr()
    original = destination.read_bytes()
    assert main([*args("data-check"), "--selection", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["selection_hash"]
    validation_args = args("data-check")
    validation_args[-1] = "validation"
    assert main([*validation_args, "--selection", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["role"] == "validation"
    with pytest.raises(SystemExit) as error:
        main([*args("make-splits"), "--output", str(destination)])
    assert error.value.code == 1
    assert destination.read_bytes() == original


@pytest.mark.parametrize(
    "payload",
    [
        '{"q-001": [], "q-001": []}',
        '{"q-001": {"a": 1}}',
        '{"q-001": [{"doc_id":"a","score":1,"extra":2}]}',
    ],
)
def test_cli_rankings_reject_duplicate_keys_and_wrong_shape(tmp_path, payload):
    path = tmp_path / "rankings.json"
    path.write_text(payload)
    with pytest.raises(SystemExit) as error:
        main([*args("evaluate"), "--rankings", str(path)])
    assert error.value.code == 1


def test_cli_fixture_evaluation_reports_missing_query(capsys):
    ranking = files("atlasir").joinpath("fixtures", "beir", "rankings.json")
    assert main([*args("evaluate"), "--rankings", str(ranking)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["coverage"]["missing_queries"] == ["q-002"]
    assert result["metrics"]["MRR@10"] == 0.625
