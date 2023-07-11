from pydantic import BaseModel

from scripts.apiserver import TTS
from tortoise.utils.audio import load_audio, float2pcm
import os
import numpy as np
from scipy.io.wavfile import write as write_wav
import wave
import pyaudio
import time
from functools import wraps


def generate(text, preset, reference_clips):
    tts = TTS.get()
    pcm_audio = tts.tts_with_preset(
        text,
        voice_samples=reference_clips,
        preset=preset,
        cvvp_amount=0.0,
        half=True,
        latent_averaging_mode=0,
        sampler="p", # ["p", "dpm++2m", "ddim"]
        diffusion_iterations=10,
        cond_free=True,
        high_vram=False
    )
    return pcm_audio


class GenerateCommand(BaseModel):
    voices_root: str
    text: str
    preset: str
    voice: str

    def __call__(self):
        reference_clips = None
        if self.voice != "random":
            voice_dir = os.path.join(self.voices_root, self.voice)
            files = os.listdir(voice_dir)
            clips_paths = [os.path.join(voice_dir, f) for f in files]
            print(f"Files: {clips_paths}")
            reference_clips = [load_audio(p, 22050) for p in clips_paths]

        pcm_audio = generate(self.text, self.preset, reference_clips)
        pcm_floats = pcm_audio.cpu().numpy()
        pcm_data = float2pcm(pcm_floats)
        write_wav("tortoise2_generation.wav", 22050, pcm_data)
        with open("tortoise2_generation.wav", "rb") as inp:
            buf = inp.read()
        return buf
