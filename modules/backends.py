"""
Transcription engines.

Each backend exposes:
    transcribe(audio_path, settings, callbacks) -> (segments, info)
where callbacks is a TranscribeCallbacks with:
    progress(done_seconds, total_seconds)   # called as segments arrive
    segment(segment_dict)                   # live transcript
    status(message)                         # human readable phase text
    should_stop() -> bool
    wait_if_paused()                        # blocks while paused
"""
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from utils.ffmpeg_utils import find_binary
from utils.cuda_utils import setup_cuda
from utils import childproc

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class TranscribeCallbacks:
    progress: Callable[[float, float], None] = lambda d, t: None
    segment: Callable[[Dict], None] = lambda s: None
    status: Callable[[str], None] = lambda m: None
    should_stop: Callable[[], bool] = lambda: False
    wait_if_paused: Callable[[], None] = lambda: None
    set_process: Callable[[Optional[subprocess.Popen]], None] = lambda p: None
    extra: Dict = field(default_factory=dict)


class StoppedError(Exception):
    pass


# ----------------------------------------------------------------------------
# faster-whisper (CTranslate2)
# ----------------------------------------------------------------------------
_model_cache = {}
_model_lock = threading.Lock()


def faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def cuda_available(cuda_lib_dir: str = "") -> bool:
    try:
        setup_cuda(cuda_lib_dir)
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _resolve_device(device: str, cuda_lib_dir: str = "") -> str:
    if device == "auto":
        return "cuda" if cuda_available(cuda_lib_dir) else "cpu"
    if device == "cuda":
        setup_cuda(cuda_lib_dir)
    return device


def model_repo_id(model: str) -> str:
    """Size name -> Hugging Face repo id used by faster-whisper (passes repo ids through)"""
    try:
        from faster_whisper.utils import _MODELS
        return _MODELS.get(model, model)
    except Exception:
        return model


def is_model_cached(model: str, model_dir: str = "") -> bool:
    """True if the faster-whisper model is already in the local cache"""
    if os.path.isdir(model) and os.path.isfile(os.path.join(model, "model.bin")):
        return True
    repo = model_repo_id(model)
    try:
        from huggingface_hub import try_to_load_from_cache
        res = try_to_load_from_cache(repo, "model.bin", cache_dir=model_dir or None)
        return isinstance(res, str) and os.path.isfile(res)
    except Exception:
        return False


def download_model(model: str, model_dir: str = "") -> str:
    """Fetch a faster-whisper model into the cache; returns the local folder"""
    from faster_whisper import download_model as _dl
    return _dl(model, cache_dir=model_dir or None)


def _load_model(settings: Dict, cb: TranscribeCallbacks):
    from faster_whisper import WhisperModel

    device = _resolve_device(settings["device"], settings.get("cuda_lib_dir", ""))
    compute = settings["compute_type"]
    if compute == "default":
        compute = "float16" if device == "cuda" else "int8"
    key = (settings["model"], device, compute, settings.get("model_dir", ""), int(settings.get("cpu_threads", 0)))
    with _model_lock:
        model = _model_cache.get(key)
        if model is None:
            _model_cache.clear()  # keep at most one model resident
            cb.status(f"Loading model {settings['model']} on {device} ({compute}) — downloading on first use…")
            kwargs = dict(device=device, compute_type=compute)
            if settings.get("model_dir"):
                kwargs["download_root"] = settings["model_dir"]
            if int(settings.get("cpu_threads", 0)) > 0:
                kwargs["cpu_threads"] = int(settings["cpu_threads"])
            model = WhisperModel(settings["model"], **kwargs)
            _model_cache[key] = model
    return model, device, compute


def transcribe_faster_whisper(audio_path: str, settings: Dict, cb: TranscribeCallbacks) -> Tuple[List[Dict], Dict]:
    model, device, compute = _load_model(settings, cb)
    if cb.should_stop():
        raise StoppedError()

    language = None if settings["language"] == "auto" else settings["language"]
    kwargs = dict(
        language=language,
        task=settings["task"],
        beam_size=int(settings["beam_size"]),
        vad_filter=bool(settings["vad_filter"]),
        word_timestamps=bool(settings["word_timestamps"]),
        condition_on_previous_text=bool(settings["condition_on_previous_text"]),
    )
    if settings["vad_filter"]:
        kwargs["vad_parameters"] = dict(min_silence_duration_ms=int(settings["vad_min_silence_ms"]))
    if settings.get("initial_prompt"):
        kwargs["initial_prompt"] = settings["initial_prompt"]
    if settings.get("hotwords"):
        # terms to weight the decoder towards, without a prompt's side effects: this is what stops
        # "harvest fair" being decoded as the commoner words it half sounds like
        kwargs["hotwords"] = settings["hotwords"]
    if settings.get("temperature") is not None:
        # a verification pass has to be an independent draw: at temperature 0 the same settings give the
        # same output, and decoding the file twice would prove nothing
        kwargs["temperature"] = float(settings["temperature"])
    spans = settings.get("clip_spans") or []
    if spans:
        # only these stretches of audio, for a pass that is re-checking what earlier passes disagreed on
        kwargs["clip_timestamps"] = [float(t) for span in spans for t in (span[0], span[1])]
        kwargs["vad_filter"] = False              # the windows are the selection now
        kwargs.pop("vad_parameters", None)

    cb.status("Transcribing…")
    segments_iter, info = model.transcribe(audio_path, **kwargs)
    total = float(info.duration or 0)
    segments = []
    for seg in segments_iter:
        cb.wait_if_paused()
        if cb.should_stop():
            raise StoppedError()
        d = {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        # what the decoder thought of its own output: the second pass and the hallucination checks need it
        for field in ("avg_logprob", "no_speech_prob", "compression_ratio", "temperature"):
            value = getattr(seg, field, None)
            if value is not None:
                d[field] = round(float(value), 4)
        if seg.words:
            d["words"] = [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in seg.words]
        segments.append(d)
        cb.segment(d)
        cb.progress(d["end"], total)
    meta = {"engine": "faster-whisper", "model": settings["model"], "device": device,
            "compute_type": compute, "language": info.language,
            "language_probability": round(float(info.language_probability or 0), 3),
            "duration": total}
    return segments, meta


# ----------------------------------------------------------------------------
# whisper.cpp (external whisper-cli)
# ----------------------------------------------------------------------------
_WCPP_LINE = re.compile(r"^\[(\d+):(\d+):(\d+)[.,](\d+)\s*-->\s*(\d+):(\d+):(\d+)[.,](\d+)\]\s*(.*)$")
_WCPP_PROGRESS = re.compile(r"progress\s*=\s*(\d+)%")


def find_whisper_cli(override: str = "") -> Optional[str]:
    for name in ("whisper-cli", "whisper-cpp", "whisper", "main"):
        found = find_binary(name, override)
        if found and name != "main":
            return found
    return find_binary("whisper-cli", override)


def _wcpp_model_path(settings: Dict) -> str:
    """Resolve ggml model file: explicit path in 'model' or <model_dir>/ggml-<size>.bin"""
    model = settings["model"]
    if os.path.isfile(model):
        return model
    candidates = []
    model_dir = settings.get("model_dir") or ""
    for d in filter(None, [model_dir, os.getcwd(), os.path.dirname(os.path.abspath(sys.argv[0])),
                           os.path.join(os.path.expanduser("~"), ".cache", "whisper.cpp")]):
        candidates += [os.path.join(d, f"ggml-{model}.bin"), os.path.join(d, f"{model}.bin"),
                       os.path.join(d, "models", f"ggml-{model}.bin")]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"whisper.cpp model 'ggml-{model}.bin' not found. Set the model folder on the Model tab "
        f"(download models with whisper.cpp's models/download-ggml-model.sh) or enter a full path to the .bin file.")


def _ts_to_seconds(h, m, s, frac) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac) / (10 ** len(frac))


def transcribe_whisper_cpp(audio_path: str, settings: Dict, cb: TranscribeCallbacks) -> Tuple[List[Dict], Dict]:
    exe = find_whisper_cli(settings.get("whisper_cli_path", ""))
    if not exe:
        raise FileNotFoundError("whisper-cli executable not found. Set its path on the Advanced tab.")
    model_path = _wcpp_model_path(settings)
    total = float(cb.extra.get("duration") or 0)

    cmd = [exe, "-m", model_path, "-f", audio_path, "-pp", "-np"]
    # -np suppresses banner/timings on stderr; timestamps stay on stdout
    cmd += ["-l", "auto" if settings["language"] == "auto" else settings["language"]]
    if settings["task"] == "translate":
        cmd.append("-tr")
    if int(settings["beam_size"]) > 1:
        cmd += ["-bs", str(int(settings["beam_size"]))]
    if int(settings.get("cpu_threads", 0)) > 0:
        cmd += ["-t", str(int(settings["cpu_threads"]))]
    if settings.get("initial_prompt"):
        cmd += ["--prompt", settings["initial_prompt"]]
    if settings.get("temperature") is not None:
        cmd += ["-tp", str(float(settings["temperature"]))]
    if settings.get("device") == "cpu":
        cmd.append("-ng")
    if settings.get("extra_args"):
        cmd += shlex.split(settings["extra_args"])

    cb.status("Transcribing with whisper.cpp…")
    proc = childproc.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           encoding="utf-8", errors="replace", bufsize=1)
    cb.set_process(proc)
    stderr_lines = []

    def _drain_err():
        for line in proc.stderr:
            stderr_lines.append(line)
            m = _WCPP_PROGRESS.search(line)
            if m and total:
                cb.progress(total * int(m.group(1)) / 100.0, total)
    t = threading.Thread(target=_drain_err, daemon=True)
    t.start()

    segments = []
    try:
        for line in proc.stdout:
            if cb.should_stop():
                childproc.kill(proc)
                raise StoppedError()
            m = _WCPP_LINE.match(line.strip())
            if not m:
                continue
            start = _ts_to_seconds(*m.groups()[0:4])
            end = _ts_to_seconds(*m.groups()[4:8])
            text = m.group(9).strip()
            if not text:
                continue
            d = {"start": start, "end": end, "text": text}
            segments.append(d)
            cb.segment(d)
            if total:
                cb.progress(end, total)
        proc.wait()
    finally:
        childproc.forget(proc)
        cb.set_process(None)
        t.join(timeout=2)
    if cb.should_stop():
        raise StoppedError()
    if proc.returncode != 0:
        tail = "".join(stderr_lines[-15:]).strip()
        raise RuntimeError(f"whisper-cli exited with code {proc.returncode}:\n{tail}")
    meta = {"engine": "whisper.cpp", "model": os.path.basename(model_path),
            "language": settings["language"], "duration": total}
    return segments, meta


# ---------------------------------------------------------------------------
# Parakeet Redux (Photon runtime) — external, in its own venv
# ---------------------------------------------------------------------------
# Attribution: the weights are Moondream's CC-BY-4.0 quantisation of NVIDIA's
# `parakeet-tdt-0.6b-v3` (also CC-BY-4.0); the runtime that runs them —
# `moondream` + `kestrel-kernels` (M87 Labs) — is PROPRIETARY, licensed only
# under a separate agreement, and is never bundled or redistributed by this app.
# `photon_runner.py`, which is what this module drives, is MIT (vlc-ai-subs).
# The full list is in the README's "Credits and licences".
# Where vlc-ai-subs installs the runtime (venv-photon + photon_runner.py). Both
# can be pointed elsewhere with the environment, and a batch user who never
# installed vlc-ai-subs can install the runtime on its own.
REDUX_ROOT = os.path.join(os.path.expanduser("~"), ".local", "share", "vlc-ai-subs")
REDUX_VENV_ENV = "WHISPERER_REDUX_VENV"
REDUX_RUNNER_ENV = "WHISPERER_REDUX_RUNNER"
REDUX_INSTALL_HINT = ("Install it with vlc-ai-subs' installer "
                      "(VSCL_AISUBS_PHOTON=1 ./install-photon-model.sh), or point "
                      f"{REDUX_VENV_ENV} and {REDUX_RUNNER_ENV} at an existing runtime.")
# The variant names the runtime itself understands (VSCL_AISUBS_PHOTON_MODEL)
REDUX_MODELS = ("redux", "ultra")
DEFAULT_REDUX_MODEL = "redux"
RAW_SEGMENTS_ENV = "VSCL_AISUBS_RAW_SEGMENTS"
# Per-word spans come back in the same call (`timestamps="word"`), so asking for
# them costs a query parameter, not a second decode. The engine asks when the
# worker's cue work needs words — snapping, resync, the sentence repairs — and maps
# what arrives into the segment dicts `utils/subtitle_utils.py` reads. Measured on
# 40 s of film dialogue, the spans are what makes those repairs fire at all: 48
# cue-text lines differ against the same run with the request suppressed.
WORDS_ENV = "VSCL_AISUBS_PHOTON_WORDS"


def redux_venv_dir(venv_dir: str = "") -> str:
    return venv_dir or os.environ.get(REDUX_VENV_ENV) or os.path.join(REDUX_ROOT, "venv-photon")


def redux_python(venv_dir: str = "") -> Optional[str]:
    """The engine venv's interpreter, or None when the runtime is not installed"""
    root = redux_venv_dir(venv_dir)
    for name in ("python3", "python"):
        path = os.path.join(root, "bin", name)
        if os.path.isfile(path):
            return path
    win = os.path.join(root, "Scripts", "python.exe")           # a Windows install
    return win if os.path.isfile(win) else None


def redux_runner(runner_path: str = "") -> Optional[str]:
    path = (runner_path or os.environ.get(REDUX_RUNNER_ENV)
            or os.path.join(REDUX_ROOT, "photon_runner.py"))
    return path if os.path.isfile(path) else None


def redux_available(venv_dir: str = "", runner_path: str = "") -> bool:
    return bool(redux_python(venv_dir) and redux_runner(runner_path))


def redux_model_id(model: str) -> str:
    """The runtime's variant name, a literal Hugging Face repo id, or the default.

    `VSCL_AISUBS_PHOTON_MODEL` takes `redux` / `ultra` **or a literal repo id** —
    anything else is taken as an id, so a Whisper size left over from another
    engine (`large-v3`, `recommended`) would be fetched as a repo and fail with
    "Photon failed: …" after the audio was already decoded. Those map to the
    measured default instead; the model dropdown is editable, so a real repo id
    still gets through.
    """
    key = (model or "").strip()
    if key.lower() in REDUX_MODELS:
        return key.lower()
    return key if "/" in key else DEFAULT_REDUX_MODEL


def transcribe_parakeet_redux(audio_path: str, settings: Dict, cb) -> Tuple[List[Dict], Dict]:
    python, runner = redux_python(), redux_runner()
    if not python or not runner:
        raise FileNotFoundError(
            "Parakeet Redux engine not installed (needs the Photon runtime: PyTorch + moondream in "
            f"their own venv, plus photon_runner.py). {REDUX_INSTALL_HINT}")
    if settings.get("task") == "translate":
        raise RuntimeError("Parakeet Redux transcribes only — it has no translate head. "
                           "Use faster-whisper for the translate task.")

    model = redux_model_id(settings.get("model", ""))
    asked = (settings.get("model") or "").strip()
    if asked and asked.lower() not in REDUX_MODELS and "/" not in asked:
        cb.status(f"Parakeet Redux: '{asked}' is not a variant of this engine — running {model}.")
    total = float(cb.extra.get("duration") or 0)
    language = "auto" if settings.get("language", "auto") == "auto" else settings["language"]

    env = dict(os.environ)
    env["PYTHONPATH"] = ""                       # a foreign PYTHONPATH loads another venv's torch
    env["VSCL_AISUBS_DEVICE"] = settings.get("device") or "auto"
    env["VSCL_AISUBS_PHOTON_MODEL"] = model
    env[RAW_SEGMENTS_ENV] = "1"                  # this app applies its own cue rules
    if settings.get("word_timestamps"):
        # snapping, resync and the sentence repairs align to words; the runtime returns
        # them in the same call (`timestamps="word"`), so this costs a query parameter
        # rather than a second decode
        env[WORDS_ENV] = "1"
    threads = int(settings.get("cpu_threads", 0) or 0)
    if threads > 0:
        # the runtime sizes its own thread pool otherwise; OpenMP is the knob torch's
        # CPU kernels read (ponytail: OMP only — add MKL_NUM_THREADS if a box shows no effect)
        env["OMP_NUM_THREADS"] = str(threads)

    cmd = [python, runner, audio_path, model, language, "transcribe"]
    cb.status("Transcribing with Parakeet Redux (one decode — segments arrive when it finishes)…")
    proc = childproc.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           encoding="utf-8", errors="replace", bufsize=1, env=env)
    cb.set_process(proc)
    stderr_lines: List[str] = []

    def _drain_err():
        for line in proc.stderr:
            stderr_lines.append(line)
            if len(stderr_lines) > 200:          # keep the tail only: this can be a torch warning flood
                del stderr_lines[:100]

    t = threading.Thread(target=_drain_err, daemon=True)
    t.start()

    segments: List[Dict] = []
    error = None
    try:
        for line in proc.stdout:
            if cb.should_stop():
                childproc.kill(proc)
                raise StoppedError()
            cb.wait_if_paused()
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            if kind == "status":
                cb.status(str(event.get("msg") or ""))
            elif kind == "sub":
                text = (event.get("text") or "").strip()
                if not text:
                    continue
                d = {"start": float(event["start"]), "end": float(event["end"]), "text": text}
                words = [{"start": float(w["start"]), "end": float(w["end"]), "word": w["word"]}
                         for w in (event.get("words") or []) if (w.get("word") or "").strip()]
                if words:
                    d["words"] = words          # the shape the app's cue work reads
                segments.append(d)
                cb.segment(d)
                if total:
                    cb.progress(d["end"], total)
            elif kind == "error":
                error = str(event.get("msg") or "unknown error")
        proc.wait()
    finally:
        childproc.forget(proc)
        cb.set_process(None)
        t.join(timeout=2)

    if cb.should_stop():
        raise StoppedError()
    if error:
        raise RuntimeError(f"Parakeet Redux: {error}")
    if proc.returncode != 0:
        tail = "".join(stderr_lines[-15:]).strip()
        raise RuntimeError(f"photon_runner exited with code {proc.returncode}:\n{tail}")

    meta = {"engine": "Parakeet Redux (Photon)", "model": model, "device": env["VSCL_AISUBS_DEVICE"],
            "language": settings.get("language", "auto"), "duration": total}
    return segments, meta


BACKENDS = {
    "faster_whisper": transcribe_faster_whisper,
    "whisper_cpp": transcribe_whisper_cpp,
    "parakeet_redux": transcribe_parakeet_redux,
}
