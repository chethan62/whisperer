"""The release's licence texts and notices (build.py).

A binary redistribution carries obligations the source tree does not: LGPL-3.0 (Qt)
and Apache-2.0 (tokenizers, huggingface_hub, hf-xet) want the licence text with the
binary, and MPL-2.0 (tqdm) wants the source's whereabouts told. Python wheels ship a
one-line licence field and no text, PySide6 included, so the texts are committed in
`licences/` and build.py copies them in — these tests pin that they are still there,
still the right licences, and that every bundled package gets a notice.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build  # noqa: E402

# first meaningful line of each text, so a truncated or swapped file fails loudly
EXPECTED_TEXTS = {
    "LGPL-3.0.txt": "GNU LESSER GENERAL PUBLIC LICENSE",
    "GPL-3.0.txt": "GNU GENERAL PUBLIC LICENSE",
    "Apache-2.0.txt": "Apache License",
}


def _first_line(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return line.strip()
    return ""


def _collected(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "BUILD_DIR", str(tmp_path / "build"))
    return build.licence_files()


def test_the_required_licence_texts_are_present_and_intact(tmp_path, monkeypatch):
    folder = _collected(tmp_path, monkeypatch)
    for name, first_line in EXPECTED_TEXTS.items():
        path = os.path.join(folder, name)
        assert os.path.isfile(path), f"{name} is required with the binary but is missing"
        assert os.path.getsize(path) > 2000, f"{name} looks truncated"
        assert first_line.lower() in _first_line(path).lower().replace("  ", " "), _first_line(path)


def test_every_bundled_package_gets_a_notice_with_a_declared_licence(tmp_path, monkeypatch):
    """A package that ships in the binary with no licence statement is the failure
    this guards: the wheel's own metadata is the only place the licence is stated."""
    folder = _collected(tmp_path, monkeypatch)
    index = open(os.path.join(folder, "INDEX.txt"), encoding="utf-8").read()
    for name in build.BUNDLED:
        try:
            __import__("importlib.metadata", fromlist=["distribution"]).distribution(name)
        except Exception:
            continue                     # not installed here: not in this build either
        assert f"  {name} " in index.replace("  PySide6 ", "  PySide6 ") or name in index, name
        assert "not declared by the wheel" not in index, f"{name} ships with no licence statement"
    notices = [f for f in os.listdir(folder) if f.endswith("-notice.txt")]
    assert notices, "no notices were generated at all"


def test_the_index_names_the_copyleft_components(tmp_path, monkeypatch):
    """LGPL-3.0 and MPL-2.0 change what the release may do; a reader of the notice
    file has to be able to see them without opening 17 wheels."""
    index = open(os.path.join(_collected(tmp_path, monkeypatch), "INDEX.txt"), encoding="utf-8").read()
    assert "LGPL-3.0" in index, "Qt is LGPL-3.0 and its text ships here"
    assert "MPL-2.0" in index, "tqdm is MPL-2.0: weak copyleft, source whereabouts required"
    assert "LICENSE file next to the executable" in index, "the app's own MIT licence is shipped too"


def test_the_texts_copied_match_the_committed_ones(tmp_path, monkeypatch):
    """build.py copies, it does not download: the release carries the reviewed text."""
    folder = _collected(tmp_path, monkeypatch)
    for name in EXPECTED_TEXTS:
        committed = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "licences", name)
        assert open(committed, encoding="utf-8").read() == open(os.path.join(folder, name), encoding="utf-8").read()


def test_the_proprietary_runtime_can_never_enter_a_release():
    """The Photon runtime (moondream + kestrel-kernels, M87 Labs) is proprietary, licensed
    only under a separate written agreement. The app deliberately does not bundle it — the
    engine drives an external venv — and the build must refuse to pick it up even if an
    import of it ever appears."""
    cmd = " ".join(build.pyinstaller_command())
    assert "--exclude-module=moondream" in cmd
    assert "--exclude-module=kestrel_kernels" in cmd
    assert not [n for n in build.BUNDLED if n.lower() in ("moondream", "kestrel-kernels", "kestrel_kernels")], \
        "and it is not in the bundled notices either — it is never redistributed"
