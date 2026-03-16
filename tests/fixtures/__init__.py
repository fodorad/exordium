"""Test fixture paths."""

import pathlib

FIXTURES_ROOT = pathlib.Path(__file__).parent

AUDIO_MULTISPEAKER = FIXTURES_ROOT / "audio" / "multispeaker.wav"
IMAGE_CAT_TIE = FIXTURES_ROOT / "image" / "cat_tie.jpg"
IMAGE_MULTISPEAKER = FIXTURES_ROOT / "image" / "multispeaker.png"
IMAGE_FACE = FIXTURES_ROOT / "image" / "face.jpg"
IMAGE_EMMA = FIXTURES_ROOT / "image" / "emma.jpg"
VIDEO_MULTISPEAKER = FIXTURES_ROOT / "video" / "multispeaker.mp4"
VIDEO_MULTISPEAKER_SHORT = FIXTURES_ROOT / "video" / "multispeaker_short.mp4"
