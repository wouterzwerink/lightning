import os
import os.path as osp
import sys
from typing import Callable
from unittest.mock import MagicMock

import lightning_app
from lightning_app.frontend import JustPyFrontend
from lightning_app.frontend.just_py import just_py
from lightning_app.frontend.just_py.just_py_base import main, webpage


def render_fn(get_state: Callable) -> Callable:
    return webpage


def test_justpy_frontend(monkeypatch):

    justpy = MagicMock()
    popen = MagicMock()
    monkeypatch.setitem(sys.modules, "justpy", justpy)
    monkeypatch.setattr(just_py, "Popen", popen)

    frontend = JustPyFrontend(render_fn=render_fn)
    flow = MagicMock()
    flow.name = "c"
    frontend.flow = flow
    frontend.start_server("a", 90)

    path = osp.join(osp.dirname(lightning_app.frontend.just_py.__file__), "just_py_base.py")

    assert popen._mock_call_args[0][0] == f"{sys.executable} {path}"
    env = popen._mock_call_args[1]["env"]
    assert env["LIGHTNING_FLOW_NAME"] == "c"
    assert env["LIGHTNING_RENDER_FUNCTION"] == "render_fn"
    assert env["LIGHTNING_HOST"] == "a"
    assert env["LIGHTNING_PORT"] == "90"

    monkeypatch.setattr(os, "environ", env)

    main()

    assert justpy.app._mock_mock_calls[0].args[0] == "/c"
    assert justpy.app._mock_mock_calls[0].args[1] == webpage

    assert justpy.justpy._mock_mock_calls[0].args[0] == webpage
    assert justpy.justpy._mock_mock_calls[0].kwargs == {"host": "a", "port": 90}
