"""Which audio stream gets transcribed.

The bug this pins: `-map 0:a:0` took the first audio stream, and on a MULTi release
that is the dub — an English run transcribed French. One guard, where every engine
routes through it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LANGUAGES                             # noqa: E402
from utils import ffmpeg_utils                           # noqa: E402


def streams(*tags):
    return [{"index": i, "language": t, "title": ""} for i, t in enumerate(tags)]


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
