from tortoise import api
from tortoise.models.vocoder import VocConf


class TTS:
    tts = None

    @staticmethod
    def get(high_vram=True, vocoder=VocConf.Univnet):
        if TTS.tts is None:
            TTS.tts = api.TextToSpeech(
                high_vram=high_vram,
                vocoder=vocoder
            )
        return TTS.tts

    @staticmethod
    def reset():
        TTS.tts = None
