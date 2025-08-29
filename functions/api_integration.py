import os
from dotenv import load_dotenv
from openai import OpenAI
import json

class LLMClient(OpenAI):
    """
        An OpenAI-compatible client that can be configured to work with different
        LLM providers based on a JSON configuration file.

        This class inherits from openai.OpenAI and configures the base_url and api_key
        during initialization, allowing it to function as a standard OpenAI client
        for the specified provider.
    """

    def __init__(self, provider_name: str, config_path: str = r"D:\pycharm\pythonProject5\api.json"):
        """
        Initializes the client for a specific provider.
        Args:
            provider_name (str): The name of the provider (e.g., "ollama", "openai").
                                 This must correspond to a key in the config file.
            config_path (str): The path to the JSON configuration file.

        Raises:
            ValueError: If the configuration file is not found, the provider is not
                        in the config, or a required API key is not set.
        """
        try:
            with open(config_path, "r") as file:
                config_data = json.load(file)
        except FileNotFoundError:
            raise ValueError(f"Json файл не найден: {config_path}")
        if provider_name not in config_data:
            raise ValueError(f"{provider_name} не найден в списке провайдеров")
        provider_info = config_data[provider_name]
        base_url = provider_info.get("base_url")
        api_key = " "
        if provider_info.get("requires_api_key"):
            api_key_var = provider_info.get("api_key_env_var")
            if not api_key_var:
                raise ValueError(f"{provider_name} не найден в списке API ключей")
            api_key = os.getenv(api_key_var)
            if not api_key:
                raise ValueError(f"{api_key_var} не задан или задан некорректно '{provider_name}'.")
        super().__init__(base_url=base_url, api_key=api_key)