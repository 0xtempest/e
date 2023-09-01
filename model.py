from pydantic import BaseModel

class InputData(BaseModel):
    system_prompt: str
    message: str
    conversation: list