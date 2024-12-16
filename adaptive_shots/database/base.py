from abc import ABC, abstractmethod
from typing import Optional, Tuple
from ..models.schemas import ShotPromptsList, RegisterPromptModel

class BaseDatabase(ABC):
    """Abstract base class defining the interface for shot databases."""

    @abstractmethod
    def register_prompt(self, prompt: str, answer: str, rating: float,
                       domain: str, used_shots: Optional[ShotPromptsList] = None) -> None:
        """Register a new prompt with its answer and rating."""
        pass

    @abstractmethod
    def get_best_shots(self, prompt: str, domain: str, c: float, limit: int) -> ShotPromptsList:
        """Retrieve the best matching prompts based on the specified criteria."""
        pass

    @abstractmethod
    def create_one_shot_prompt(self, prompt: str, domain: str) -> Tuple[str, Optional[ShotPromptsList]]:
        """Create a one-shot prompt using the most relevant record."""
        pass

    @abstractmethod
    def create_few_shots_prompt(self, prompt: str, domain: str, limit: int) -> Tuple[str, ShotPromptsList]:
        """Create a few-shot prompt by combining multiple relevant records."""
        pass
