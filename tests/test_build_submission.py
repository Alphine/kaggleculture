import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import build_submission


def test_split_imports_handles_multiline_parenthesized_import():
    source = (
        "from .config import (\n"
        "    A,\n"
        "    B,\n"
        ")\n"
        "import math\n"
        "\n"
        "X = 1\n"
    )
    external_imports, body = build_submission.split_imports_and_body(source)
    assert external_imports == ["import math"]
    assert "from .config" not in body
    assert "import math" not in body
    assert "X = 1" in body


def test_build_produces_compilable_main(tmp_path, monkeypatch):
    output = tmp_path / "main.py"
    monkeypatch.setattr(build_submission, "OUTPUT", output)
    build_submission.build()
    content = output.read_text(encoding="utf-8")
    compile(content, str(output), "exec")
    assert "def agent(obs)" in content
