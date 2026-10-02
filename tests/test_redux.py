"""The Parakeet Redux engine: registration, install discovery, and its contract.

The runtime is not installed in CI (it is a ~1.6 GB PyTorch venv behind an opt-in
installer), so the engine is exercised against a stand-in `photon_runner.py` that
speaks the same JSONL contract. What that proves is the part this repo owns: the
command line, the environment the runner is handed, and how its events become
segments, progress and errors.
"""
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ENGINES  # noqa: E402
from modules import backends  # noqa: E402

_STATUS = '{"type": "status", "msg": "Photon parakeet-redux (en, transcribe, cpu, native ternary kernel)"}'
_ENVLINE = ('{"type": "status", "msg": "RAW=%s MODEL=%s DEVICE=%s PYTHONPATH=%s"}')


def _fake_runtime(tmp_path, body: str, monkeypatch):
    """A venv with an executable 'python3' and a photon_runner.py next to it.

    `body` is the sh script the fake interpreter runs, so each test can pick what
    the runner says (cues, an error event, a silent non-zero exit).
    """
    venv = tmp_path / "venv-photon"
    (venv / "bin").mkdir(parents=True)
    interpreter = venv / "bin" / "python3"
    interpreter.write_text("#!/bin/sh\n" + body)
    interpreter.chmod(interpreter.stat().st_mode | stat.S_IEXEC)
    runner = tmp_path / "photon_runner.py"
    runner.write_text("# stand-in for the real runner\n")
    monkeypatch.setenv(backends.REDUX_VENV_ENV, str(venv))
    monkeypatch.setenv(backends.REDUX_RUNNER_ENV, str(runner))
    return interpreter, runner


class _Cb:
    """Records what the worker would have shown the user."""

    def __init__(self, duration=5.0):
        self.segments, self.progress, self.statuses = [], [], []
        self.killed, self.extra = None, {"duration": duration}
        self.stop, self.paused = False, 0

    def progress_cb(self, done, total):
        self.progress.append((done, total))

    def segment(self, seg):
        self.segments.append(seg)

    def status(self, msg):
        self.statuses.append(msg)

    def should_stop(self):
        return self.stop

    def wait_if_paused(self):
        self.paused += 1

    def set_process(self, proc):
        self.killed = proc


def _callbacks(cb):
    return backends.TranscribeCallbacks(progress=cb.progress_cb, segment=cb.segment, status=cb.status,
                                        should_stop=cb.should_stop, wait_if_paused=cb.wait_if_paused,
                                        set_process=cb.set_process, extra=cb.extra)


CUE_BODY = "\n".join([
    f'echo {_STATUS!r}',
    'echo "{\\"type\\": \\"status\\", \\"msg\\": \\"RAW=$VSCL_AISUBS_RAW_SEGMENTS MODEL=$VSCL_AISUBS_PHOTON_MODEL '
    'DEVICE=$VSCL_AISUBS_DEVICE PYTHONPATH=$PYTHONPATH\\"}"',
    'echo \'{"type": "sub", "i": 1, "start": 0.5, "end": 2.5, "text": " Hello there. "}\'',
    'echo \'{"type": "sub", "i": 2, "start": 3.0, "end": 5.0, "text": "General Kenobi."}\'',
    'echo \'{"type": "done", "segments": 2, "srt_path": null}\'',
]) + "\n"


def _settings(**over):
    s = {"engine": "parakeet_redux", "model": "redux", "device": "auto", "language": "en", "task": "transcribe"}
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# Wiring: a dropdown entry with no backend is the classic silent failure
# ---------------------------------------------------------------------------

def test_the_engine_is_offered_and_registered():
    assert "parakeet_redux" in ENGINES, "the Engine dropdown is built from config.ENGINES"
    assert "parakeet_redux" in backends.BACKENDS, "and it must have an implementation"
    assert backends.BACKENDS["parakeet_redux"] is backends.transcribe_parakeet_redux


def test_the_variant_mapping_never_invents_a_model():
    assert backends.redux_model_id("redux") == "redux"
    assert backends.redux_model_id("ultra") == "ultra"
    assert backends.redux_model_id("recommended") == "recommended"   # the runtime's own default
    assert backends.redux_model_id("large-v3") == "redux", "a left-over Whisper size is not a variant"
    assert backends.redux_model_id("") == "redux"


# ---------------------------------------------------------------------------
# The contract with the runner
# ---------------------------------------------------------------------------

def test_cues_progress_and_the_environment_reach_the_caller(tmp_path, monkeypatch):
    _fake_runtime(tmp_path, CUE_BODY, monkeypatch)
    cb = _Cb(duration=5.0)
    segments, meta = backends.transcribe_parakeet_redux(str(tmp_path / "audio.wav"), _settings(), _callbacks(cb))

    assert [s["text"] for s in segments] == ["Hello there.", "General Kenobi."], "text is stripped"
    assert segments[0]["start"] == 0.5 and segments[1]["end"] == 5.0
    assert cb.segments == segments, "every cue reaches the live transcript as it arrives"
    assert cb.progress == [(2.5, 5.0), (5.0, 5.0)], "progress follows the cue ends, against the media duration"

    envline = [s for s in cb.statuses if s.startswith("RAW=")][0]
    assert "RAW=1" in envline, "this app applies its own cue rules, so the runner must not"
    assert "MODEL=redux" in envline
    assert "DEVICE=auto" in envline, "device=auto is resolved by the runner's own rule, not here"
    assert envline.endswith("PYTHONPATH="), "a foreign PYTHONPATH loads another venv's torch"
    assert cb.killed is None, "the child is unregistered once it is gone"

    assert meta["engine"] == "Parakeet Redux (Photon)" and meta["model"] == "redux"
    assert meta["duration"] == 5.0


def test_the_chosen_variant_is_passed_to_the_runtime(tmp_path, monkeypatch):
    _fake_runtime(tmp_path, CUE_BODY, monkeypatch)
    cb = _Cb()
    backends.transcribe_parakeet_redux(str(tmp_path / "a.wav"), _settings(model="ultra"), _callbacks(cb))
    assert any("MODEL=ultra" in s for s in cb.statuses)


def test_an_error_event_becomes_the_exception(tmp_path, monkeypatch):
    _fake_runtime(tmp_path, 'echo \'{"type": "error", "msg": "Photon failed: no kernel"}\'\nexit 1\n', monkeypatch)
    cb = _Cb()
    with pytest.raises(RuntimeError) as exc:
        backends.transcribe_parakeet_redux(str(tmp_path / "a.wav"), _settings(), _callbacks(cb))
    assert "no kernel" in str(exc.value), "the runner's own message is the actionable one"


def test_a_silent_non_zero_exit_reports_the_stderr_tail(tmp_path, monkeypatch):
    _fake_runtime(tmp_path, 'echo "torch: cannot load libcudnn" >&2\nexit 3\n', monkeypatch)
    cb = _Cb()
    with pytest.raises(RuntimeError) as exc:
        backends.transcribe_parakeet_redux(str(tmp_path / "a.wav"), _settings(), _callbacks(cb))
    message = str(exc.value)
    assert "code 3" in message and "libcudnn" in message


def test_a_missing_runtime_names_the_installer(tmp_path, monkeypatch):
    monkeypatch.setenv(backends.REDUX_VENV_ENV, str(tmp_path / "nope"))
    monkeypatch.setenv(backends.REDUX_RUNNER_ENV, str(tmp_path / "nope" / "photon_runner.py"))
    assert backends.redux_available() is False
    cb = _Cb()
    with pytest.raises(FileNotFoundError) as exc:
        backends.transcribe_parakeet_redux(str(tmp_path / "a.wav"), _settings(), _callbacks(cb))
    assert "install-photon-model.sh" in str(exc.value)


def test_translate_is_refused_before_anything_is_spawned(tmp_path, monkeypatch):
    _fake_runtime(tmp_path, "exit 0\n", monkeypatch)
    cb = _Cb()
    with pytest.raises(RuntimeError) as exc:
        backends.transcribe_parakeet_redux(str(tmp_path / "a.wav"), _settings(task="translate"), _callbacks(cb))
    assert "translate" in str(exc.value)
    assert cb.statuses == [], "nothing was started, so nothing should have been announced"
