import os

import pytest
from click.testing import CliRunner
from tests_app import _PROJECT_ROOT

from lightning_app.cli.lightning_cli import show
from lightning_app.testing.testing import run_app_in_cloud, wait_for


@pytest.mark.cloud
def test_boring_app_example_cloud() -> None:
    with run_app_in_cloud(
        os.path.join(_PROJECT_ROOT, "examples/app_boring/"),
        app_name="app_dynamic.py",
        debug=True,
    ) as (
        _,
        view_page,
        _,
        name,
    ):

        def check_hello_there(*_, **__):
            locator = view_page.frame_locator("iframe").locator('ul:has-text("Hello there!")')
            if len(locator.all_text_contents()):
                return True

        wait_for(view_page, check_hello_there)

        runner = CliRunner()
        result = runner.invoke(show.commands["logs"], [name])
        lines = result.output.splitlines()

        assert result.exit_code == 0
        assert result.exception is None
        assert any("--filepath=/content/.storage/boring_file.txt" in line for line in lines)
        print("Succeeded App!")
