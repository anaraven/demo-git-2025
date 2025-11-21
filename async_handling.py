from abc import ABC, abstractmethod
import asyncio
from enum import Enum
import random
import time
import os

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
import matplotlib.pyplot as plt
from openai import AsyncOpenAI, OpenAI

# Load API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print('Set GOOGLE_API_KEY in your environment to run live calls.')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("Set OPENAI_API_KEY in the environment")

class Provider(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"

class GenerativeAIClient(ABC):
    @abstractmethod
    async def generate_async(self, prompt:str)->str:
        ...
    @abstractmethod
    def generate_sync(self, prompt: str) -> str:
        ...
    @property
    @abstractmethod
    def name(self) -> str:
        ...

class GeminiAIClient(GenerativeAIClient):
    def __init__(self, api_key:str):
        self.api_key = api_key

    # Attribute needed for clearer logging
    @property
    def name(self)->str:
        return "Gemini"
    async def generate_async(self, prompt: str) -> str:
        client = genai.Client(api_key=self.api_key, 
                              http_options=HttpOptions(api_version="v1"))
        config = GenerateContentConfig(temperature=0.4)
        response = await client.aio.models.generate_content(model="gemini-2.5-flash", 
                                                            contents=prompt, 
                                                            config=config)
        if not response or not response.text:
            raise ValueError("Empty response from Gemini")
        return response.text  # type: ignore
    def generate_sync(self, prompt: str) -> str:
        client = genai.Client(api_key=self.api_key, 
                              http_options=HttpOptions(api_version="v1"))
        config = GenerateContentConfig(temperature=0.4)
        response = client.models.generate_content(model="gemini-2.5-flash", 
                                                  contents=prompt, 
                                                  config=config)
        if not response or not response.text:
            raise ValueError("Empty response from Gemini")
        return response.text

    
class OpenAIClient(GenerativeAIClient):
    def __init__(self, api_key:str):
        self.api_key = api_key

    ## Attribute needed for clearer logging
    @property
    def name(self)->str:
        return "Openai"
    async def generate_async(self, prompt: str) -> str:
        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(model="gpt-4o", 
                                                        messages=[{"role": "user", "content": prompt}], 
                                                        temperature=0.4)
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty response from OpenAI")
        return response.choices[0].message.content
    def generate_sync(self, prompt: str) -> str:
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(model="gpt-4o", 
                                                        messages=[{"role": "user", "content": prompt}], 
                                                        temperature=0.4)
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty response from OpenAI")
        return response.choices[0].message.content


class GenerativeAIClientFactory:
    @staticmethod
    def create(provider:Provider, api_key:str) -> GenerativeAIClient:
        if provider == Provider.GEMINI:
            return GeminiAIClient(api_key)
        elif provider == Provider.OPENAI:
            return OpenAIClient(api_key)
        else:
            raise ValueError(f"Unknown provider {provider}")

def log_event(timeline: list, task: str, event: str, initial_timestamp: float) -> None:
    """
    Log an event with timestamp to visualize when operations start and end
    You can simply ignore this this does not serve any purpose in the pipeline.
    Args:
        timeline: List to append the event to
        task: Name/ID of the task
        event: Description of what happened (e.g., START or END)
    """
    timestamp = time.monotonic() - initial_timestamp  # Calculate time since start
    timeline.append({
        "task": task,
        "event": event,
        "timestamp": timestamp,
    })
    print(f"[{timestamp:.2f}] {event}: {task}")


async def async_db_operation(step:str, async_timeline_events: list, initial_timestamp: float):
    """
    Simulate a database write operation (like saving analytics or logs)
    
    In real applications, this would be writing data to a database,
    which is I/O-bound and benefits from asynchronous execution.
    """
    task_id = f"log_{step}"
    log_event(async_timeline_events, task_id, "START", initial_timestamp) # just logging time
    # Simulate variable database operation time
    await asyncio.sleep(random.uniform(0.3, 3.0))
    log_event(async_timeline_events, task_id, "END", initial_timestamp) # just logging time

async def async_llm_call(prompt, generative_ai_client:GenerativeAIClient, async_timeline_events: list, initial_timestamp: float):
    """
    Simulate a call to a Large Language Model API
    
    In real applications, this would be a network request to an API like
    OpenAI, Anthropic, or a self-hosted LLM service.
    """
    
    task_id = f"{generative_ai_client.name}_{prompt}"
    log_event(async_timeline_events, task_id, "START", initial_timestamp) # just logging time

    response = await generative_ai_client.generate_async(prompt)

    # LLM calls often take longer than other API calls
    await asyncio.sleep(random.uniform(0.5, 3.2))
    log_event(async_timeline_events, task_id, "END", initial_timestamp) # just logging time
    return f"RESPONSE_{generative_ai_client.name}: {response}"

async def async_http_call(endpoint, async_timeline_events: list, initial_timestamp: float):
    """
    Simulate an HTTP request to an external API
    
    In real applications, this could be fetching data from a REST API,
    microservice, or other web service.
    """
    task_id = f"http_{endpoint.replace('/', '')}"
    log_event(async_timeline_events, task_id, "START", initial_timestamp) # just logging time
    # Simulate variable network latency
    await asyncio.sleep(random.uniform(0.4, 4.0))
    log_event(async_timeline_events, task_id, "END", initial_timestamp) # just logging time
    return f"HTTP result from {endpoint}"

async def run_async_pipeline(async_timeline_events: list, initial_timestamp: float):
    """
    Main workflow that coordinates the execution of different async operations
    
    This demonstrates both:
    1. Dependent tasks (using 'await') - where we need results immediately
    2. Background tasks (using create_task) - where we can fire and forget
    """
    log_event(async_timeline_events, "main_pipeline", "START_PIPELINE", initial_timestamp)

    tasks = []
    # DEPENDENT TASK: We need the LLM result before proceeding
    gemini = GenerativeAIClientFactory.create(Provider.GEMINI, GOOGLE_API_KEY) # type: ignore
    gemini_response = await async_llm_call("Just say Hello World", gemini, async_timeline_events, initial_timestamp)
    print(gemini_response)
    openai = GenerativeAIClientFactory.create(Provider.OPENAI, OPENAI_API_KEY) # type: ignore
    openai_response = await async_llm_call("Just say Hello World", openai, async_timeline_events, initial_timestamp)
    print(openai_response)
    # BACKGROUND TASK: Log the result to the database, but don't wait for it
    tasks.append(asyncio.create_task(async_db_operation("gemini_result", async_timeline_events, initial_timestamp)))
    tasks.append(asyncio.create_task(async_db_operation("openai_result", async_timeline_events, initial_timestamp )))

    # DEPENDENT TASK: We need this HTTP result for our business logic
    http1_gemini = await async_http_call("/api/data", async_timeline_events, initial_timestamp)
    # BACKGROUND TASK: Log this API call, but don't block on it
    tasks.append(asyncio.create_task(async_db_operation("http1_gemini", async_timeline_events, initial_timestamp)))
    # DEPENDENT TASK: We need this HTTP result for our business logic
    http1_openai = await async_http_call("/api/data", async_timeline_events, initial_timestamp)
    # BACKGROUND TASK: Log this API call, but don't block on it
    tasks.append(asyncio.create_task(async_db_operation("http1_openai", async_timeline_events, initial_timestamp)))



    # DEPENDENT TASK: We need this data too before proceeding
    http2_gemini = await async_http_call("/api/details", async_timeline_events, initial_timestamp)
    # BACKGROUND TASK: Another non-blocking logging operation
    tasks.append(asyncio.create_task(async_db_operation("http2_gemini", async_timeline_events, initial_timestamp)))
    # DEPENDENT TASK: We need this data too before proceeding
    http2_openai = await async_http_call("/api/details", async_timeline_events, initial_timestamp)
    # BACKGROUND TASK: Another non-blocking logging operation
    tasks.append(asyncio.create_task(async_db_operation("http2_openai", async_timeline_events, initial_timestamp)))

    # DEPENDENT TASK: Generate a summary using the LLM with collected data
    summary_gemini = await async_llm_call("Summarize the prompt",gemini, async_timeline_events, initial_timestamp)
    # BACKGROUND TASK: Log the summary generation
    tasks.append(asyncio.create_task(async_db_operation("summary_gemini", async_timeline_events, initial_timestamp)))
    # DEPENDENT TASK: Generate a summary using the LLM with collected data
    summary_openai = await async_llm_call("Summarize everything",openai, async_timeline_events, initial_timestamp)
    # BACKGROUND TASK: Log the summary generation
    tasks.append(asyncio.create_task(async_db_operation("summary_openai", async_timeline_events, initial_timestamp)))
    
    # Wait for all background tasks to complete before exiting
    # This ensures all logging operations finish properly
    # This will run without gathering all tasks as well, you can try
    await asyncio.gather(*tasks)
    log_event(async_timeline_events, "main_pipeline", "END_PIPELINE", initial_timestamp)


def sync_db_operation(step, sync_timeline_events: list, initial_timestamp):
    """
    Simulate a synchronous database write operation
    
    In real applications, this would be writing data to a database,
    but here we're blocking the entire program while it runs.
    """
    task_id = f"log_{step}"
    log_event(sync_timeline_events, task_id, "START", initial_timestamp)
    # Block the entire program during this "database operation"
    time.sleep(random.uniform(0.3, 3.0))
    log_event(sync_timeline_events, task_id, "END", initial_timestamp)

def sync_llm_call(prompt, generative_ai_client:GenerativeAIClient, sync_timeline_events: list, initial_timestamp: float):
    """
    Simulate a synchronous call to a Large Language Model API
    
    In real applications, this would be waiting for a response from
    an AI service like OpenAI or Anthropic, completely blocking execution.
    """
    task_id = f"{generative_ai_client.name}_{prompt}"
    log_event(sync_timeline_events, task_id, "START", initial_timestamp)
    # Block the entire program during this "API call"
    time.sleep(random.uniform(0.5, 3.2))
    response = generative_ai_client.generate_sync(prompt)
    log_event(sync_timeline_events, task_id, "END", initial_timestamp)
    return f"RESPONSE_{generative_ai_client.name}: {response}"

def sync_http_call(endpoint, sync_timeline_events: list, initial_timestamp: float):
    """
    Simulate a synchronous HTTP request to an external API
    
    In real applications, this would be waiting for a network response,
    with the program doing nothing else during this time.
    """
    task_id = f"http_{endpoint.replace('/', '')}"
    log_event(sync_timeline_events, task_id, "START", initial_timestamp)
    # Block the entire program during this "network request"
    time.sleep(random.uniform(0.4, 4.0))
    log_event(sync_timeline_events, task_id, "END", initial_timestamp)
    return f"HTTP result from {endpoint}"

def run_sync_pipeline(sync_timeline_events: list, initial_timestamp: float):
    """
    Main workflow that coordinates operations in a strictly sequential manner
    
    Each operation completely blocks the program until it finishes,
    resulting in poor resource utilization and longer total execution time.
    """
    log_event(sync_timeline_events, "main_pipeline", "START_PIPELINE", initial_timestamp)

    # Step 1: Make LLM call and wait for it to complete
    gemini = GenerativeAIClientFactory.create(Provider.GEMINI, GOOGLE_API_KEY) # type: ignore
    gemini_response = sync_llm_call("Just say Hello World", gemini, sync_timeline_events, initial_timestamp)
    print(gemini_response)
    # Step 2: Log the result and wait for logging to complete
    sync_db_operation("gemini_result", sync_timeline_events, initial_timestamp)
    # Step 1: Make LLM call and wait for it to complete
    openai = GenerativeAIClientFactory.create(Provider.OPENAI, OPENAI_API_KEY) # type: ignore
    openai_response = sync_llm_call("Just say Hello World", openai, sync_timeline_events, initial_timestamp)
    print(openai_response)
    # Step 2: Log the result and wait for logging to complete
    sync_db_operation("openai_result", sync_timeline_events, initial_timestamp)

    # Step 3: Make first HTTP call and wait for it to complete
    http1_gemini = sync_http_call("/api/data", sync_timeline_events, initial_timestamp)
    http1_openai = sync_http_call("/api/data", sync_timeline_events, initial_timestamp)
    # Step 4: Log the HTTP result and wait for logging to complete
    sync_db_operation("http1_gemini", sync_timeline_events, initial_timestamp)
    sync_db_operation("http1_openai", sync_timeline_events, initial_timestamp)

    # Step 5: Make second HTTP call and wait for it to complete
    http2_gemini = sync_http_call("/api/details", sync_timeline_events, initial_timestamp)
    http2_openai = sync_http_call("/api/details", sync_timeline_events, initial_timestamp)
    # Step 6: Log the HTTP result and wait for logging to complete
    sync_db_operation("http2_gemini", sync_timeline_events, initial_timestamp)
    sync_db_operation("http2_openai", sync_timeline_events, initial_timestamp)

    # Step 7: Make final LLM call and wait for it to complete
    summary_gemini = sync_llm_call("Summarize the prompt",gemini, sync_timeline_events, initial_timestamp)
    summary_openai = sync_llm_call("Summarize the prompt",openai, sync_timeline_events, initial_timestamp)
    # Step 8: Log the summary and wait for logging to complete
    sync_db_operation("summary_gemini", sync_timeline_events, initial_timestamp)
    sync_db_operation("summary_openai", sync_timeline_events, initial_timestamp)

    log_event(sync_timeline_events, "main_pipeline", "END_PIPELINE", initial_timestamp)

def plot_timelines(async_timeline_events: list, sync_timeline_events: list):
  """Plot the timelines for async and sync operations"""

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

  # Extract async events to get start times and durations
  async_data = {}
  for event in async_timeline_events:
      task = event['task']
      if task not in async_data:
          async_data[task] = {}
      if event['event'] in ['START', 'START_PIPELINE']:
          async_data[task]['start'] = event['timestamp']
      elif event['event'] in ['END', 'END_PIPELINE']:
          async_data[task]['end'] = event['timestamp']

  # Filter only complete tasks (have both start and end)
  async_tasks = [t for t in async_data.keys() if 'start' in async_data[t] and 'end' in async_data[t]]
  async_starts = [async_data[t]['start'] for t in async_tasks]
  async_durations = [async_data[t]['end'] - async_data[t]['start'] for t in async_tasks]

  # Async chart
  ax1.barh(
      async_tasks,
      async_durations,
      left=async_starts,  
      align='center',
      color='skyblue'
  )
  ax1.set_yticks(range(len(async_tasks)))
  ax1.set_yticklabels(async_tasks)
  ax1.set_ylabel("Async Tasks")
  ax1.set_title("Async Execution Timeline")
  ax1.invert_yaxis()
  ax1.grid(axis='x', linestyle='--', alpha=0.6)

  # extract Sync events for chart
  sync_data = {}
  for event in sync_timeline_events:
      task = event['task']
      if task not in sync_data:
          sync_data[task] = {}
      if event['event'] in ['START', 'START_PIPELINE']:
          sync_data[task]['start'] = event['timestamp']
      elif event['event'] in ['END', 'END_PIPELINE']:
          sync_data[task]['end'] = event['timestamp']

  sync_tasks = [t for t in sync_data.keys() if 'start' in sync_data[t] and 'end' in sync_data[t]]
  sync_starts = [sync_data[t]['start'] for t in sync_tasks]
  sync_durations = [sync_data[t]['end'] - sync_data[t]['start'] for t in sync_tasks]

  # Sync chart
  ax2.barh(
      sync_tasks,
      sync_durations,
      left=sync_starts,  # This makes them start at correct time!
      align='center',
      color='lightcoral'
  )
  ax2.set_yticks(range(len(sync_tasks)))
  ax2.set_yticklabels(sync_tasks)
  ax2.set_ylabel("Sync Tasks")
  ax2.set_xlabel("Time (s)")
  ax2.set_title("Sync Execution Timeline")
  ax2.invert_yaxis()
  ax2.grid(axis='x', linestyle='--', alpha=0.6)

  plt.tight_layout()
  plt.show()

def main():
  print("\n--- Running Async Pipeline ---\n")
  async_timeline_events = []
  initial_timestamp = time.monotonic()
  asyncio.run(run_async_pipeline(async_timeline_events, initial_timestamp))
  print('########################')

  print("\n--- Running Sync Pipeline ---\n")
  initial_timestamp = time.monotonic()
  sync_timeline_events = [] 
  run_sync_pipeline(sync_timeline_events, initial_timestamp)
  print('########################')

  plot_timelines(async_timeline_events, sync_timeline_events)


if __name__ == "__main__":
    main()
