import sys
import logging
import toml

from client_factory import GenerativeAIClientFactory, Provider

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    try:
      with open('config.toml', 'r') as f:
          config = toml.load(f)
    except Exception as e:
      logging.error(f"Failed to load config.toml: {e}")
      config = {}

    # we do not know if logging section exists 
    logging_config = config.get('logging', {})
    assert logging_config is not None
    # here we know that there is a logging section, but level may not exist

    log_level = logging_config.get('level', 'INFO').upper()
    assert log_level is not None

    logging.getLogger().setLevel(log_level)
    logging.warning(f"Log level set to {log_level}")

    if len(sys.argv) < 2:
        logging.error("Usage: python generative_ai_adaptor.py <provider>")
        logging.error("Provider options: openai, google")
        return

    provider_str = sys.argv[1].lower()
    logging.info(f"Selected provider: {provider_str}")

    try:
        provider = Provider(provider_str)
    except ValueError:
        logging.error(f"Invalid provider: {provider_str}")
        logging.error("Provider options: openai, google")
        return

    client = GenerativeAIClientFactory.create_client(provider)
    question = "Hello, how are you?"
    logging.debug(f"Question: {question}")
    response = client.generate(question)
    logging.debug(f"Response: {response}")
    print(response)

if __name__ == "__main__":
    main()
