from typing import List
from pydantic import BaseModel, Field
from typing import Annotated

class RegisterPromptModel(BaseModel):
    """Schema for registering a new prompt with its answer and rating."""
    prompt: str
    answer: str
    rating: Annotated[float, Field(
        strict=True,
        ge=1.0,
        le=10.0,
        description="Rating must be between 1 and 10"
    )]
    domain: str

class ShotPrompt(BaseModel):
    """Schema for a single shot prompt with its metadata."""
    id: int
    prompt: str
    answer: str
    domain: str
    predicted_rating: float
    ucb: float

    def __str__(self) -> str:
        return f"Prompt: {self.prompt}\n\nAnswer: {self.answer}"

class ShotPromptsList(BaseModel):
    """Collection of shot prompts with utility methods."""
    prompts: List[ShotPrompt]

    def to_messages(self) -> List[dict]:
        """Convert prompts to a format suitable for message chains."""
        return [
            message for shot in self.prompts 
            for message in [
                {"role": "user", "content": shot.prompt},
                {"role": "assistant", "content": shot.answer}
            ]
        ]