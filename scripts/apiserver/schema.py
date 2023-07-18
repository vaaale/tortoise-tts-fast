from pydantic import BaseModel


class SpeakInput(BaseModel):
    voice: str = "alex"
    preset: str = "ultra_fast"
    text: str = "This is a test to see if I sound like the real me. Hi, my name is Alex"
