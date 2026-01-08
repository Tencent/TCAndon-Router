import re
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from prompt import router_prompt
from llm_client import completion
from utils import load_config
from datasets import load_dataset
from typing import Optional


def router(history: str, agents: str) -> str:
    """
    Route the query to the appropriate agent

    :param history: User query
    :param agents: List of agents
    :return: Router response
    """
    message = [{"role": "user", "content": router_prompt.format(agents=agents) + 'user:' + history}]
    res = completion(message)
    return res


def extract_prediction(response: str) -> list[str]:
    """
    Extract prediction ID from router response

    :param response: Router response
    :return: List of prediction IDs
    """
    agent_ids = re.findall(r'<ID>(.*?)</ID>', response)
    return agent_ids


def _eval_sample(args: tuple[str, str, str]) -> tuple[list[str], bool, Optional[dict]]:
    """
    Evaluate a single sample

    :param args: Tuple of query, label, and agents
    :return: Tuple of prediction IDs, correctness, and debug info
    """
    query, label, agents = args
    router_result = router(query, agents)
    prediction = extract_prediction(router_result)

    is_correct = label in prediction

    debug_info = None
    if not is_correct:
        debug_info = {
            "query": query,
            "label": label,
            "prediction": prediction,
            "router_output": router_result,
        }

    return prediction, is_correct, debug_info


def metric(label: list[str], prediction: list[str]) -> tuple[float, float]:
    """
    Calculate precision and recall

    :param label: List of true labels
    :param prediction: List of predicted labels
    :return: Tuple of precision and recall
    """
    label = set(label)
    prediction = set(prediction)
    tp = len(label & prediction)

    precision = tp / (len(prediction) + 1e-10)
    recall = tp / len(label)
    return precision, recall


def evaluate_dataset(dataset_name: str, limit: Optional[int] = None, processes: Optional[int] = None) -> None:
    """
    Evaluate the specified dataset

    :param dataset_name: Name of the dataset
    :param limit: Limit number of samples to evaluate
    :param processes: Number of processes to use
    """
    os.makedirs('data', exist_ok=True)
    # Load test data
    df = load_dataset("Router/router_data", cache_dir='data')['test'].to_pandas()

    # Filter by dataset name
    df = df[df["dataset"] == dataset_name]
    # Limit samples if specified
    if limit:
        df = df[:limit]
    print(f"Evaluating {len(df)} samples from {dataset_name}")
    if len(df) == 0:
        print(f"No samples found for dataset: {dataset_name}")
        return
    query = df['query'].values
    label = df['intent'].values

    agents = load_config(f'config/{dataset_name}_config.xml')

    tasks = [(q, l, agents) for q, l in zip(query, label)]

    predictions = []
    correct_state = []
    correct = 0

    # Set number of processes
    if processes is None:
        processes = cpu_count()

    # Use multiprocessing pool to evaluate samples
    with Pool(processes=processes) as pool:
        for pred_list, is_correct, debug_info in tqdm(
                pool.imap(_eval_sample, tasks),
                total=len(tasks),
                desc=f"Evaluating {dataset_name}"
        ):
            predictions.append(pred_list)
            correct_state.append(is_correct)
            if is_correct:
                correct += 1
            else:
                # Print incorrect predictions
                print('*' * 100)
                print("Incorrect prediction")
                print(f"Query: {debug_info['query']}")
                print(f"True: {debug_info['label']}, Predicted: {debug_info['prediction']}")
                print(f"Router output:\n{debug_info['router_output']}")

    accuracy = correct / len(query)
    print(f"Dataset {dataset_name} Accuracy: {accuracy}")

    df['prediction'] = predictions
    df['correct'] = correct_state
    df.to_csv(f'data/{dataset_name}_result.csv', index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate router performance on different datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., clinc150, hwu64, minds14, sgd)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to evaluate')
    parser.add_argument('--processes', type=int, default=1,
                        help='Number of processes to use')

    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.limit, args.processes)


if __name__ == '__main__':
    main()
