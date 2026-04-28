import json

from click.testing import CliRunner

from strands.cli import main


def test_cli_encode():
    runner = CliRunner()
    result = runner.invoke(main, ["encode", "happy dog"])
    assert result.exit_code == 0
    assert ":" in result.output


def test_cli_compare():
    runner = CliRunner()
    result = runner.invoke(main, ["compare", "happy", "joyful"])
    assert result.exit_code == 0
    score = float(result.output.strip())
    assert score > 0.8


def test_cli_inspect():
    runner = CliRunner()
    result = runner.invoke(main, ["inspect", "happy"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["found"] is True
    assert payload["domain"] == "EM"


def test_cli_inspect_missing():
    runner = CliRunner()
    result = runner.invoke(main, ["inspect", "xyzzyqqq"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["found"] is False
