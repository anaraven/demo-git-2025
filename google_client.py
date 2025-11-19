import os
from dataclasses import dataclass

from abstract_client import GenerativeAIClient
from google import genai
from google.genai.types import GenerateContentConfig

@dataclass
class GoogleGenAIClient(GenerativeAIClient):
  model: str='gemini-2.5-flash'
  temperature: float=0.7
  max_tokens: int=2000
  retries: int=3
  backoff: float=0.8

  def __post_init__(self):
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

