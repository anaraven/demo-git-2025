from enum import Enum
import logging

from abstract_client import GenerativeAIClient

class Provider(Enum):
    OPENAI = 'openai'
    GOOGLE = 'google'

class GenerativeAIClientFactory:
    
    @staticmethod
    def create_client(provider: Provider) -> GenerativeAIClient:
        if provider == Provider.OPENAI:
            logging.info("Creating OpenAI client")
            from openai_client import OpenAIClient
            return OpenAIClient()
        elif provider == Provider.GOOGLE:
            logging.info("Creating Google GenAI client")
            from google_client import GoogleGenAIClient
            return GoogleGenAIClient()
        else:
            raise ValueError(f"Unknown provider: {provider}")

