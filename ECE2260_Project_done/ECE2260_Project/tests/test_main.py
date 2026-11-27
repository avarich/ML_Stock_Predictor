from src.main import main, run_experiment


def test_full_run_on_csv(nasdaq_csv, tmp_path, capsys):
    out = tmp_path / "out"
    result = run_experiment("TEST", nasdaq_csv, 0.8, out)
    text = capsys.readouterr().out
    for expected in ["Raw data range", "Features (9)", "Train:", "Test :", "Baseline"]:
        assert expected in text
    assert (out / "TEST_report.txt").exists()
    assert (out / "TEST_accuracy_by_quarter.png").exists()
    assert result["train"]["Date"].max() < result["test"]["Date"].min()


def test_main_command_line(nasdaq_csv, tmp_path, capsys):
    main(
        [
            "--ticker",
            "TEST",
            "--csv",
            str(nasdaq_csv),
            "--output-dir",
            str(tmp_path / "o"),
        ]
    )
    assert "TEST next-day direction experiment" in capsys.readouterr().out
