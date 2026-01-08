import toml
from openai import OpenAI


def load_llm_config(config_path: str = "config/llm_config.toml") -> dict:
    """
    load llm config

    :param config_path: llm config path
    :return: llm config dict
    """
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return config['llm']


def create_llm_client(config_path: str = "config/llm_config.toml") -> tuple[OpenAI, dict]:
    """
    create llm client

    :param config_path: llm config path
    :return: llm client and llm config dict
    """
    config = load_llm_config(config_path)

    client = OpenAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    return client, config


def completion(messages: list[dict[str, str]], config_path: str = "config/llm_config.toml") -> str:
    """
    llm completion

    :param messages: messages
    :param config_path: llm config path
    :return: llm response
    """
    client, config = create_llm_client(config_path)

    response = client.chat.completions.create(
        model=config['model'],
        messages=messages,
        max_tokens=config['max_tokens'],
        temperature=config['temperature']
    )

    return response.choices[0].message.content
