"""
Dumps the conditioning latents for the specified voice to disk. These are expressive latents which can be used for
other ML models, or can be augmented manually and fed back into Tortoise to affect vocal qualities.
"""
import os
from pathlib import Path
from typing import Literal, Union

import torch
from simple_parsing.decorators import main

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, BUILTIN_VOICES_DIR

EXAMPLE = Path(__file__ + "/../../results/conditioning_latents").resolve()


@main
def main(
    voice: str = "pat2",
    voices_dir: Path = Path(BUILTIN_VOICES_DIR),
    output_path: Union[None, Path] = None,
    latent_averaging_mode: Literal[0, 1, 2] = 0,
):
    """Dumps the conditioning latents for the specified voice to disk. These are expressive latents which can be used for
    other ML models, or can be augmented manually and fed back into Tortoise to affect vocal qualities.
    Args:
        voice: Selects the voice to convert to conditioning latents
        voices_dir: Path to voices
        output_path: Where to store outputs. If set to None the sample_dir will be used
        latent_averaging_mode: How to average voice latents, 0 for standard, 1 for per-sample, 2 for per-minichunk
    """

    tts = TextToSpeech()
    voices = {voice: os.path.join(voices_dir, voice) for voice in os.listdir(voices_dir)}
    selected_voices = voice.split(",")
    for voice in selected_voices:
        cond_path = voices[voice]
        clips_paths = [os.path.join(cond_path, f) for f in os.listdir(cond_path)]
        _cp = "\n\t".join(clips_paths)
        print(f'Generating conditioning latents for {voice}:\n\t{_cp}')
        voice_samples = [load_audio(p, 22050) for p in clips_paths]

        latents = tts.get_conditioning_latents(
            voice_samples,
            return_mels=True,
            latent_averaging_mode=latent_averaging_mode,
            original_tortoise=False,
        )

        if output_path is None:
            output_path = Path(cond_path)
        else:
            output_path.mkdir(exist_ok=True)

        print(f"Latents for {voice} saved to {output_path}")
        torch.save(latents, output_path / Path(f"{voice}.pth"))


if __name__ == "__main__":
    main()
