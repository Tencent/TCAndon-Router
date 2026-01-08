from llm_client import completion
from prompt import router_prompt, refine_prompt
from utils import load_config
from utils import find_agent_id


def router(query: str, agents: str) -> str:
    """
    router function
    :param query: query
    :param agents: agents
    :return: intent
    """
    prompt = router_prompt.format(agents=agents) + 'user:' + query
    message = [{'role': 'user', 'content': prompt}]
    response = completion(message)
    return response


def agent(query: str, intent: str) -> str:
    return 'your agent'


def refine(query: str, result: str) -> None:
    """
    refine function
    :param query: query
    :param result: result
    """
    intent_list = find_agent_id(result)
    answer_str = ''
    for intent in intent_list:
        print('executing intent: ', intent)
        agent_result = agent(query, intent)
        answer_str += f'# The answer of {intent}\n{agent_result}\n'
    prompt = refine_prompt.format(query=query, answer=answer_str)
    message = [{'role': 'user', 'content': prompt}]
    response = completion(message)
    print(response)


if __name__ == '__main__':
    agents = load_config('config/hwu64_config.xml')
    query = "Can you recommend any pub in mg road"
    result = router(query, agents)
    print(result)
    refine(query, result)
