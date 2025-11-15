import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print('⚠️ Set OPENAI_API_KEY in your environment to run live calls.')
else:
    print('✅ OPENAI_API_KEY is set.')

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print('⚠️ Set GOOGLE_API_KEY in your environment to run live calls.')
else:
    print('✅ GOOGLE_API_KEY is set.')

import abc

class GenerativeAIClient(abc.ABC):

  @abc.abstractmethod
  def generate(self, prompt: str, **kwargs) -> str:
      ...

from openai import OpenAI

class OpenAIClient(GenerativeAIClient):

  def generate(self, prompt: str, *, model: str='gpt-4o-mini', temperature: float=0.7, max_tokens: int=200, retries: int=3, backoff: float=0.8) -> str:
      """Call the chat completion API with basic retries and timing.
      Returns the model's answer as plain text.
      """
      if not OPENAI_API_KEY:
          raise ValueError("OPENAI_API_KEY is not set in the environment")

      client = OpenAI(api_key=OPENAI_API_KEY)
      response = client.chat.completions.create(
          model=model,
          messages=[
              {"role": "user", "content": prompt}
          ],
          temperature=temperature,
          max_tokens=max_tokens
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

  def generate(self, prompt: str, *, model: str='gemini-2.5-flash', temperature: float=0.7, max_tokens: int=200, retries: int=3, backoff: float=0.8) -> str:
              
      """Call the chat completion API with basic retries and timing.
      Returns the model's answer as plain text.
      """

      if not isinstance(prompt, str):
          raise ValueError("Prompt should be a string")

      client = genai.Client()
      config=GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens)

      response = client.models.generate_content(model=model, contents=prompt, config=config)
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
