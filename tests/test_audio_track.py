"""Which audio stream gets transcribed.

The bug this pins: `-map 0:a:0` took the first audio stream, and on a MULTi release
that is the dub — an English run transcribed French. One guard, where every engine
routes through it.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LANGUAGES                             # noqa: E402
from utils import ffmpeg_utils                           # noqa: E402


def streams(*tags):
    """(language, title) pairs, or bare language tags"""
    out = []
    for i, t in enumerate(tags):
        lang, title = t if isinstance(t, tuple) else (t, "")
        out.append({"index": i, "language": lang, "title": title,
                    "non_dialogue": any(m in title.lower() for m in ffmpeg_utils.NON_DIALOGUE_MARKERS)})
    return out


def test_the_tag_table_covers_every_language_the_dropdown_offers():
    """A language with no entry falls back to a bare 2-letter match, which most
    containers do not write — that is a silent wrong-track transcription."""
    missing = [code for code in LANGUAGES if code != "auto" and code not in ffmpeg_utils.LANGUAGE_TAGS]
    assert not missing, f"no stream tags known for {missing}"


def test_a_multilingual_file_gets_the_requested_language():
    fr_eng = streams("fre", "eng")
    assert ffmpeg_utils.choose_audio_stream(fr_eng, "en") == (1, "2/2 · eng")
    assert ffmpeg_utils.choose_audio_stream(fr_eng, "fr") == (0, "1/2 · fre"), "/B codes count too"
    deu = streams("deu", "eng", "fra")
    assert ffmpeg_utils.choose_audio_stream(deu, "fr") == (2, "3/3 · fra"), "/T codes count too"
    assert ffmpeg_utils.choose_audio_stream(streams("eng", "ger"), "de") == (1, "2/2 · ger")


def test_an_absent_or_unknown_language_keeps_the_first_stream_and_says_so():
    """No silent wrong track: the old behaviour survives, the label reports it."""
    assert ffmpeg_utils.choose_audio_stream(streams("fre", "eng"), "ja") == (0, "1/2 · fre")
    assert ffmpeg_utils.choose_audio_stream(streams("", "eng"), "en") == (1, "2/2 · eng")
    assert ffmpeg_utils.choose_audio_stream(streams("und", "und"), "en") == (0, "1/2 · untagged")
    assert ffmpeg_utils.choose_audio_stream(streams("eng", "fre"), "auto") == (0, "1/2 · eng"), \
        "auto-detect has no preference"


def test_audio_description_is_never_mistaken_for_dialogue():
    """The measured case: a real release's AD track carries NO disposition flag at all,
    only title="Descriptive" — the standard flags are useless, the title is not. When
    only a descriptive track exists in that language it is still used (better a hedge
    than a failure), and the label admits it."""
    assert ffmpeg_utils.choose_audio_stream(streams(("eng", "Descriptive"), ("eng", "")), "en") == \
        (1, "2/2 · eng"), "the plain track, even though the AD one comes first"
    assert ffmpeg_utils.choose_audio_stream(
        streams(("eng", "Descriptive"), ("fra", "")), "en") == (0, "1/2 · eng (descriptive — no plain dialogue track)")
    assert ffmpeg_utils.choose_audio_stream(streams("eng", ("eng", "Commentary")), "en") == (0, "1/2 · eng"), \
        "one plain track and one commentary: the plain one, and the label says which"


def test_the_probe_flags_descriptive_tracks(tmp_path, monkeypatch):
    """The rule lives in the probe's parse, so pin the parse against real ffprobe JSON."""
    payload = {"streams": [
        {"index": 0, "tags": {"language": "fre", "title": "VFF"}},
        {"index": 1, "tags": {"language": "eng"}},
        {"index": 2, "tags": {"language": "eng", "title": "Descriptive"}},
        {"index": 3, "tags": {"language": "eng", "title": "Audio Description"}},
        {"index": 4, "tags": {"language": "spa"}, "disposition": {"visual_impaired": 1}},
        {"index": 5, "tags": {"language": "und"}},
    ]}
    monkeypatch.setattr(ffmpeg_utils, "find_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(ffmpeg_utils.childproc, "run",
                        lambda argv, **kw: type("P", (), {"stdout": json.dumps(payload)})())

    probed = ffmpeg_utils.probe_audio_streams("film.mkv")
    assert [s["non_dialogue"] for s in probed] == [False, False, True, True, True, False]
    assert [s["language"] for s in probed] == ["fre", "eng", "eng", "eng", "spa", "und"]
    assert ffmpeg_utils.choose_audio_stream(probed, "en") == (1, "2/6 · eng")
    assert ffmpeg_utils.choose_audio_stream(probed, "spa") == (4, "5/6 · spa (descriptive — no plain dialogue track)")


def test_one_track_means_no_choice_to_report():
    assert ffmpeg_utils.choose_audio_stream(streams("fre"), "en") == (0, ""), "nothing was chosen, so nothing is said"
    assert ffmpeg_utils.choose_audio_stream([], "en") == (0, ""), "and no ffprobe at all still works"


def test_extraction_maps_the_chosen_stream(tmp_path, monkeypatch):
    """The choice has to reach the ffmpeg command, not just the return value."""
    monkeypatch.setattr(ffmpeg_utils, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg_utils, "probe_audio_streams", lambda src: streams("fre", "eng"))
    cmd = {}

    class Proc:
        returncode = 0

        def communicate(self, timeout=None):
            return b"", b""

    def popen(argv, **kw):
        cmd["argv"] = argv
        return Proc()

    monkeypatch.setattr(ffmpeg_utils.childproc, "popen", popen)
    monkeypatch.setattr(ffmpeg_utils.childproc, "forget", lambda proc: None)

    label = ffmpeg_utils.extract_audio("film.mkv", str(tmp_path / "a.wav"), language="en")
    assert cmd["argv"][cmd["argv"].index("-map") + 1] == "0:a:1", "the English stream, not the dub"
    assert label == "2/2 · eng"
    assert "-vn" in cmd["argv"] and "-ar" in cmd["argv"], "the WAV contract is unchanged"
