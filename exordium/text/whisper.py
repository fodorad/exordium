import numpy as np
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import exordium.utils.decorator as D


class WhisperWrapper():

    def __init__(self, model_size: str = "distil-large-v3") -> None:
        """Whisper wrapper class.

        run on GPU with INT8
            model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        run on CPU with INT8
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        """
        # self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
            torch_dtype=torch.float16,
            device="cuda:0", # or mps for Mac devices
            model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
        )

    @D.timer_with_return
    def __call__(self, audio_path: str) -> tuple[list[dict], dict]:
        """Speech-to-Text with Whisper
        https://github.com/SYSTRAN/faster-whisper

        Args:
            audio_path (str): audio file path

        Returns:
            tuple[list[dict], dict]: segmented text prediction with start and end time in sec, and info about the detected language
        """
        #segments, info = self.model.transcribe(audio, beam_size=5)

        # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        #print(list(segments))
        #print(info)

        #for segment in segments:
        #    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

        #return segments, info
        outputs = self.pipe(
            audio_path,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
        )

        return outputs


if __name__ == "__main__":
    D.TIMING_ENABLED = True
    test_video = '/home/fodor/dev/EmotionLinMulT/data/db_processed/MEAD/M003/audio_wav/down-0-1-001.wav'
    model = WhisperWrapper()
    out = model(test_video)
    print(out)