from pathlib import Path
import torch
from tortoise import api
from tortoise.models.vocoder import VocConf
from tortoise.utils.audio import load_audio
import numpy as np
from scipy.io.wavfile import write as write_wav
import wave
import pyaudio
import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        retval = func(*args, **kwargs)
        print(f"Function took: {time.time() - start_time} secs.")
        return retval

    return wrapper


class AudioFile:
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.p.get_format_from_width(self.wf.getsampwidth()),
            channels=self.wf.getnchannels(),
            rate=self.wf.getframerate(),
            output=True
        )

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(self.chunk)
        while data != b'':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        """ Graceful shutdown """
        self.stream.close()
        self.p.terminate()


def float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def play_wav(filename):
    print("Testing audio output...")
    a = AudioFile(filename)
    a.play()
    a.close()


voice_dir = "/home/alex/Documents/voices"


def load_voice(voices_path: Path, voice: str, latent_averaging_mode=0, tts=None):
    _voice_path = (voices_path / voice)
    cond_files = list(_voice_path.glob("*.pth"))
    if len(cond_files) == 1:
        print(f"Loading latets from {cond_files[0]}")
        conditioning_latents = torch.load(cond_files[0])
        return conditioning_latents, None
    else:
        print(f"Loading samples from {_voice_path}")
        reference_clips = _voice_path.glob("*.wav")
        print(f"Files: {reference_clips}")
        voice_samples = [load_audio(str(p), 22050) for p in reference_clips]
        if tts is not None:
            latents = tts.get_conditioning_latents(
                voice_samples,
                return_mels=True,
                latent_averaging_mode=latent_averaging_mode,
                original_tortoise=False,
            )
            latents_path = _voice_path / Path(f"{voice}.pth")
            print(f"Saving latents to {latents_path}")
            torch.save(latents, latents_path)
            return latents, None
        return None, reference_clips


tts = api.TextToSpeech(
    high_vram=True,
    vocoder=VocConf.Univnet
)
conditioning_latents, reference_clips = load_voice(voices_path=Path(voice_dir), voice="alex", tts=tts)


@timer
def generate(text, preset):
    pcm_audio = tts.tts_with_preset(
        text,
        voice_samples=reference_clips,
        conditioning_latents=conditioning_latents,
        preset=preset,
        cvvp_amount=0.0,
        half=False,
        latent_averaging_mode=0,
        # sampler="p", # ["p", "dpm++2m", "ddim"]
        # diffusion_iterations=10,
        # cond_free=True,
    )
    return pcm_audio


def generate_and_play(text="This is a test", preset="ultra_fast"):
    pcm_audio = generate(text, preset)
    pcm_data = pcm_audio.cpu().numpy()
    pcm_data = float2pcm(pcm_data)
    write_wav("tortoise2_generation.wav", 22050, pcm_data)
    play_wav("tortoise2_generation.wav")


@timer
def main():
    preset = "ultra_fast"
    generate_and_play(
        text="This is a longer text to test if my voice is sounding like the real me, or if this needs more reference clips.This is a second sentence to see how Tortoise handles breaks.",
        preset=preset
    )

    preset = "very_fast"
    generate_and_play(
        text="This is a longer text to test if my voice is sounding like the real me, or if this needs more reference clips.This is a second sentence to see how Tortoise handles breaks.",
        preset=preset
    )


if __name__ == '__main__':
    main()
