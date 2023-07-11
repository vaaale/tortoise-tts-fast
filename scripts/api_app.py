import os

from fastapi import FastAPI

from scripts.apiserver.commands import GenerateCommand
from scripts.apiserver.schema import SpeakInput
from fastapi.responses import Response


voices_root = os.getenv("VOICE_ROOT", "/voices")

app = FastAPI()


@app.post("/speak", response_class=Response)
def speak(input: SpeakInput) -> Response:
    voice = input.voice
    preset = input.preset
    text = input.text

    generate = GenerateCommand(voices_root=voices_root, text=text, preset=preset, voice=voice)
    buf = generate()

    return Response(content=buf, media_type="audio/x-wav;codec=pcm;rate=22050")


