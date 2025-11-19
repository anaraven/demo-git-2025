from abc import ABC, abstractmethod

class GenerativeAIClient(ABC):

  @abstractmethod
  def generate(self, prompt: str, **kwargs) -> str:
      ...
