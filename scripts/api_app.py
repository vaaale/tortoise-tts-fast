import os

import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse, FileResponse

from scripts.apiserver.commands import GenerateCommand
from scripts.apiserver.schema import SpeakInput
from fastapi.responses import Response
from fastapi.requests import Request


voices_root = os.getenv("VOICE_ROOT", "/voices")

app = FastAPI()


@app.post("/speak", response_class=Response)
def speak(audio_input: SpeakInput) -> Response:
    voice = audio_input.voice
    preset = audio_input.preset
    text = audio_input.text

    generate = GenerateCommand(voices_root=voices_root, preset=preset, voice=voice)
    buf = generate(text=text)
    # audio/x-wav;codec=pcm;rate=22050
    return Response(content=buf.read(), media_type="audio/vnd.wav")


@app.post("/stream", response_class=StreamingResponse)
def speak(audio_input: SpeakInput) -> Response:
    voice = audio_input.voice
    preset = audio_input.preset
    text = audio_input.text

    generate = GenerateCommand(voices_root=voices_root, preset=preset, voice=voice)
    buf = generate(text=text)
    # audio/x-wav;codec=pcm;rate=22050
    return StreamingResponse(buf, media_type="audio/vnd.wav")


if __name__ == '__main__':
    uvicorn.run("scripts.api_app:app", host="0.0.0.0", port=8000, reload=True)