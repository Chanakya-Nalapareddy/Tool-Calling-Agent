import os
from openai import OpenAI
from Tool_Calling_Agent.logger import logger


class OpenAIClientManager:
    """Singleton manager for initializing and accessing the OpenAI client."""

    # Class-level variable to hold the OpenAI client instance
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            base_url = "http://10.246.250.226:12300/v1"
            api_key = "none"
            cls._client = OpenAI(base_url=base_url, api_key=api_key)
            logger.info(f"OpenAI client initialized with base_url={base_url}")
        return cls._client
