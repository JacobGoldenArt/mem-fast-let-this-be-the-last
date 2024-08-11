from langchain_core.runnables import RunnableConfig

from src.settings.app_config import AppConfig, AppDefaults


# config.py
class ConfigManager:
    """Initialize the ConfigManager with the provided defaults.

     Parameters:
     defaults (AppConfig): The default configuration settings to use.

     Returns:
     None
     """

    def __init__(self, defaults: AppConfig):
        self.defaults = defaults

    def create_config(self, **kwargs) -> RunnableConfig:
        """Create a new configuration by merging the default settings with the provided keyword arguments.

        Parameters:
        **kwargs: Additional configuration settings to be merged with the defaults.

        Returns:
        A dictionary representing the new configuration with merged settings.
        """
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return {"configurable": {**self.defaults, **filtered_kwargs}}

    def process_config(self, config: RunnableConfig) -> AppConfig:
        """Process the provided RunnableConfig and return an AppConfig instance.

               Parameters:
               config (RunnableConfig): The RunnableConfig to process.

               Returns:
               AppConfig: An instance of AppConfig with merged settings from defaults and the provided config.
        """
        configurable = config.get("configurable", {})
        return AppConfig(**{**self.defaults, **configurable})


config_manager = ConfigManager(AppDefaults)

__all__ = ["config_manager"]