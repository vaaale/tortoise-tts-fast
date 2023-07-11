from pydantic import BaseModel


class SpeakInput(BaseModel):
    voice: str = "random"
    preset: str = "ultra_fast"
    text: str
