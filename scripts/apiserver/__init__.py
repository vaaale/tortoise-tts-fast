from tortoise import api


class TTS:
    tts = None

    @staticmethod
    def get():
        if TTS.tts is None:
            TTS.tts = api.TextToSpeech()
        return TTS.tts
