import os
import re
import random
from typing import Optional
import xml.etree.ElementTree as ET


def load_config(config_path: str) -> str:
    """
    load config file

    :param config_path: str  path to config file
    :return: str  config file content
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as file:
        return file.read().strip()


def find_agent_id(result: str) -> list[str]:
    """
    find agent id from result

    :param result: str  result
    :return: list[str] agent id
    """
    intent = re.findall(r'<ID>(.*?)</ID>', result)
    return intent


def sample_agents(agent_dict: dict[str, dict], n: int, must_include_key: Optional[str] = None) -> list[dict]:
    """
    sample n agents from agent_dict and must_include_key must be in the result

    :param agent_dict: dict[str, dict]  agent dict
    :param n: int  number of agents to sample
    :param must_include_key: str  key to include in the result
    :return: list[dict] sampled agents
    """
    if must_include_key not in agent_dict or must_include_key is None:
        sampled_keys = random.sample(list(agent_dict.keys()), k=min(n, len(agent_dict)))
        return [agent_dict[k] for k in sampled_keys]
    remaining_keys = [k for k in agent_dict.keys() if k != must_include_key]
    sampled_keys = random.sample(remaining_keys, k=min(n - 1, len(remaining_keys)))
    selected_keys = [must_include_key] + sampled_keys
    selected_keys = random.sample(selected_keys, k=len(selected_keys))
    return [agent_dict[k] for k in selected_keys]


def load_all_agents(config_path: str) -> dict[str, dict]:
    """
    load all agents from config file

    :param config_path: str  path to config file
    :return: dict[str, dict] agent dict
    """
    agents_xml = load_config(config_path)
    root = ET.fromstring(agents_xml)
    agents: dict[str, dict] = {}
    for agent in root.findall("Agent"):
        name = agent.findtext("Name")
        desc = agent.findtext("Description")
        agent_id = agent.findtext("ID")
        agents[agent_id] = {
            "name": name,
            "description": desc,
            "id": agent_id
        }
    return agents


def agent_dic_to_xlm(agents: list[dict]) -> str:
    """
    convert agent dict to xml

    :param agents: list[dict] agent dict
    :return: str xml content
    """
    xml_content = "<Agents>\n"
    for agent in agents:
        xml_content += f"    <Agent>\n"
        xml_content += f"        <Name>{agent['name']}</Name>\n"
        xml_content += f"        <Description>{agent['description']}</Description>\n"
        xml_content += f"        <ID>{agent['id']}</ID>\n"
        xml_content += f"    </Agent>\n"
    xml_content += "</Agents>"
    return xml_content


def load_all_agents_xlm(config_path: str) -> str:
    """
    load all agents from config file
    :param config_path: str  path to config file
    :return: str xml content
    """
    return agent_dic_to_xlm(load_all_agents(config_path))
