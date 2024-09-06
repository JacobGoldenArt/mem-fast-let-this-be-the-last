from langchain_anthropic import ChatAnthropic
from langchain_fireworks.chat_models import ChatFireworks
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI


def _get_model(config, default, key):
    model = config["configurable"].get(key, default)
    temp = config["configurable"].get("temperature", default)
    if model == "oai-gpt4":
        return ChatOpenAI(model="gpt-4o-2024-08-06", temperature=temp, streaming=True)
    elif model == "oai-gpt4m":
        return ChatOpenAI(model="gpt-4o-mini", temperature=temp, streaming=True)
    elif model == "an-35-s":
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=temp, streaming=True)  # type: ignore
    elif model == "mis-nemo":
        return ChatMistralAI(model="open-mistral-nemo", temperature=temp, streaming=True)
    elif model == "mis-lg":
        return ChatMistralAI(model="mistral-large-latest", temperature=temp, streaming=True)
    elif model == "fw-lm31-405b":
        return ChatFireworks(
            model_name="accounts/fireworks/models/llama-v3p1-405b-instruct", temperature=temp, streaming=True)
    elif model == "fw-lm31-150b":
        return ChatFireworks(
            model_name="accounts/fireworks/models/llama-v3p1-150b-instruct", temperature=temp, streaming=True)
    elif model == "fw-lm31-8b":
        return ChatFireworks(
            model_name="accounts/fireworks/models/llama-v3p1-8b-instruct", temperature=temp, streaming=True)
    else:
        raise ValueError


__all__ = ["_get_model"]