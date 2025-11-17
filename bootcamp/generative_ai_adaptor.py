import abc
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class GenerativeAIClient(abc.ABC):

  @abc.abstractmethod
  def generate(self, prompt: str, **kwargs) -> str:
      ...

from dataclasses import dataclass
from openai import OpenAI

@dataclass
class OpenAIClient(GenerativeAIClient):
  model: str='gpt-4o-mini'
  temperature: float=0.7
  max_tokens: int=200
  retries: int=3
  backoff: float=0.8

  def __post_init__(self):

      OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
      if not OPENAI_API_KEY:
          raise ValueError("OPENAI_API_KEY is not set in the environment")

      self.client = OpenAI(api_key=OPENAI_API_KEY)

  def generate(self, prompt: str) -> str:
      """Call the chat completion API with basic retries and timing.
      Returns the model's answer as plain text.
      """
      response = self.client.chat.completions.create(
          model=self.model,
          messages=[
              {"role": "user", "content": prompt}
          ],
          temperature=self.temperature,
          max_tokens=self.max_tokens
      )
      if response is None:
          raise ValueError("No response from the API")

      choices = response.choices
      if not choices or len(choices) == 0:
          raise ValueError("Failed to get a valid response from the API")

      first_choice = choices[0]
      message = first_choice.message
      if message.role != "assistant":
          raise ValueError("Invalid message format in the response")

      if not message.content:
          reason = message.refusal
          raise ValueError("No content in the assistant's message: " + str(reason))
      
      return message.content

from google import genai
from google.genai.types import GenerateContentConfig

class GoogleGenAIClient(GenerativeAIClient):
  model: str='gemini-2.5-flash'
  temperature: float=0.7
  max_tokens: int=200
  retries: int=3
  backoff: float=0.8

  def __post_init__(self):
      try:
          from dotenv import load_dotenv
          load_dotenv()
      except Exception:
          pass

      GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
      if not GOOGLE_API_KEY:
          raise ValueError("GOOGLE_API_KEY is not set in the environment")
      self.client = genai.Client(api_key= GOOGLE_API_KEY)
      self.config=GenerateContentConfig(temperature=self.temperature, max_output_tokens=self.max_tokens)


  def generate(self, prompt: str) -> str:
              
      """Call the chat completion API with basic retries and timing.
      Returns the model's answer as plain text.
      """

      if not isinstance(prompt, str):
          raise ValueError("Prompt should be a string")

      response = self.client.models.generate_content(model=self.model, contents=prompt, config=self.config)
      if response is None:
          raise ValueError("No response from the API")
      
      if not response.text:
          if response.candidates:
              reason = response.candidates[0].finish_reason
              if reason == "MAX_TOKENS":
                  raise ValueError("Response was cut off due to max tokens limit.")
              raise ValueError(f"No content in the response: {reason}")
          raise ValueError("Failed to get a valid response from the API")
      
      return response.text


from enum import Enum

class Provider(Enum):
    OPENAI = 'openai'
    GOOGLE = 'google'

class GenerativeAIClientFactory:
    
    @staticmethod
    def create_client(provider: Provider) -> GenerativeAIClient:
        if provider == Provider.OPENAI:
            return OpenAIClient()
        elif provider == Provider.GOOGLE:
            return GoogleGenAIClient()
        else:
            raise ValueError(f"Unknown provider: {provider}")

def main():
    client = OpenAIClient()
    response = client.generate("Hello, how are you?")
    print(response)

    client = GoogleGenAIClient()
    response = client.generate("Hello, how are you?")
    print(response)

    client = GenerativeAIClientFactory.create_client(Provider.OPENAI)
    response = client.generate("Hello, how are you?")
    print(response)

if __name__ == "__main__":
    main()
