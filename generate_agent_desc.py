import re
import os
import argparse
from tqdm import tqdm
from llm_client import completion
from prompt import description_prompt
from datasets import load_dataset


def generate_desc(message: list[dict[str, str]]) -> str:
    """
    Generate agent description from message

    :param message: list of message
    :return: agent description
    """
    res = completion(message)
    try:
        match = re.search(r"<Description>(.*?)</Description>", res, re.DOTALL)
        if match:
            desc = match.group(1)
        else:
            raise ValueError("No <Description> tag found.")
        return desc.strip()
    except Exception as e:
        raise ValueError(f"XML parsing failed: {e}\nResponse: {res}")


def safe_generate_desc(message: list[dict[str, str]], max_retries: int = 3) -> str:
    """
    Safe generate agent description from message

    :param message: list of message
    :return: agent description
    """
    for attempt in range(1, max_retries + 1):
        try:
            desc = generate_desc(message)
            return desc
        except Exception as e:
            print(f"[Attempt {attempt}] Error generating desc: {e}")
    raise RuntimeError("Failed to generate description after max retries")


def main(dataset_name: str, limit: int = 30) -> None:
    df = load_dataset("joska/open-router-data", cache_dir='data')['train'].to_pandas()
    df = df[df["dataset"] == dataset_name]
    df = df[~df['intent'].str.contains('oos', na=False)]
    print(df['intent'].value_counts())

    xml_content = "<Agents>\n"
    for intent in tqdm(df['intent'].unique()):
        text = ''
        for t in df[df['intent'] == intent]['query'].values[:limit]:
            text += t + '\n'
        message = [{'role': 'user', 'content': description_prompt.format(text=text.strip())}]
        desc = safe_generate_desc(message)
        xml_content += f"    <Agent>\n"
        xml_content += f"        <Name>{intent}</Name>\n"
        xml_content += f"        <Description>{desc}</Description>\n"
        xml_content += f"        <ID>{intent}</ID>\n"
        xml_content += f"    </Agent>\n"
    xml_content += "</Agents>"

    os.makedirs(f'config', exist_ok=True)
    with open(f'config/{dataset_name}_config_test.xml', 'w', encoding='utf-8') as f:
        f.write(xml_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate agent description')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., clinc150, hwu64, minds14, sgd)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to evaluate')

    args = parser.parse_args()
    main(args.dataset, args.limit)
