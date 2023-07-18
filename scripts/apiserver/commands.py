from io import BytesIO
from pathlib import Path

import torch
from pydantic import BaseModel
from scripts.apiserver import TTS
from tortoise.utils.audio import load_audio, float2pcm
from scipy.io.wavfile import write as write_wav


class GenerateCommand(BaseModel):
    voices_root: str
    preset: str
    voice: str

    @staticmethod
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

    @staticmethod
    def generate(text, preset, reference_clips, latents):
        tts = TTS.get()
        pcm_audio = tts.tts_with_preset(
            text,
            voice_samples=reference_clips,
            conditioning_latents=latents,
            preset=preset,
            cvvp_amount=0.0,
            half=True,
            latent_averaging_mode=0,
        )
        return pcm_audio

    def __call__(self, text: str) -> BytesIO:
        latents = None
        reference_clips = None
        if self.voice != "random":
            latents, reference_clips = self.load_voice(
                voices_path=Path(self.voices_root),
                voice=self.voice,
                latent_averaging_mode=0,
                tts=TTS.get()
            )
        pcm_audio = self.generate(text, self.preset, reference_clips, latents=latents)
        pcm_floats = pcm_audio.cpu().numpy()
        pcm_data = float2pcm(pcm_floats)
        # write_wav("tortoise2_generation.wav", 22050, pcm_data)
        wav_buf = BytesIO()
        write_wav(wav_buf, 22050, pcm_data)
        # with open("tortoise2_generation.wav", "rb") as inp:
        #     buf = inp.read()
        wav_buf.seek(0)
        # return buf
        return wav_buf
