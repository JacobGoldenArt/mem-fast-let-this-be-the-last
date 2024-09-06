from typing import TypedDict, Type

from langchain_core.runnables.config import RunnableConfig, ensure_config

from src.settings.app_config import AppConfig, AppDefaults


class ConfigManager:
    """
    A utility class for managing configuration in LangGraph applications.

    This class simplifies the process of creating, processing, and using configuration
    objects that are compatible with LangChain's RunnableConfig while also providing
    type safety and default values for application-specific configurations.

    Attributes:
        config_class (Type[TypedDict]): The TypedDict class defining the structure of the application-specific configuration.
        defaults (dict): A dictionary of default values for the configuration.

    Example:
        class MyGraphConfig(TypedDict):
            model: str
            thread_id: str
            user_id: str
            temperature: float

        defaults = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
        config_manager = ConfigManager(MyGraphConfig, defaults)

        # Creating a config
        config = config_manager.create_config(thread_id="thread_123", user_id="user_456")

        # Using in an agent or node
        async def agent(state: Any, raw_config: RunnableConfig):
            config = config_manager.process_config(raw_config)
            # Use config.model, config.thread_id, etc.
            print(f"Using model: {config['model']}")
            print(f"Thread ID: {config['thread_id']}")

        # Simulating agent call
        import asyncio
        asyncio.run(agent({}, config))
        Using model: gpt-3.5-turbo
        Thread ID: thread_123
    """

    def __init__(self, config_class: Type[TypedDict], defaults: dict):
        self.config_class = config_class
        self.defaults = defaults

    def process_config(self, config: RunnableConfig) -> TypedDict:
        """
        Process a RunnableConfig into an application-specific TypedDict configuration.

        This method ensures that the input config is a valid RunnableConfig, extracts
        the 'configurable' section, and merges it with default values to create a
        fully-populated instance of the application-specific configuration.

        Args:
            config (RunnableConfig): The input configuration, typically from a LangChain Runnable.

        Returns:
            TypedDict: An instance of the application-specific configuration class.

        Example:
            raw_config = {"configurable": {"thread_id": "thread_789"}}
            processed_config = config_manager.process_config(raw_config)
            print(processed_config)
            {'model': 'gpt-3.5-turbo', 'thread_id': 'thread_789', 'temperature': 0.7}
        """
        full_config = ensure_config(config)
        configurable = full_config.get("configurable", {})
        return self.config_class(
            **{k: configurable.get(k, self.defaults.get(k)) for k in self.config_class.__annotations__}
        )

    def create_config(self, **kwargs) -> RunnableConfig:
        """
        Create a RunnableConfig with application-specific values.

        This method creates a RunnableConfig object, ensuring that all required fields
        are present by combining provided values with defaults.

        Args:
            **kwargs: Key-value pairs for configuration options.

        Returns:
            RunnableConfig: A configuration object compatible with LangChain Runnables.

        Example:
            config = config_manager.create_config(thread_id="thread_101", temperature=0.8)
            print(config)
            {'configurable': {'model': 'gpt-3.5-turbo', 'thread_id': 'thread_101', 'temperature': 0.8}}
        """
        return ensure_config({
            "configurable": {
                **self.defaults,
                **kwargs
            }
        })


config_manager = ConfigManager(AppConfig, AppDefaults)
__all__ = ["config_manager"]