# TCAndon-Router

<p align="center">
    <img src="assets/router.png" width="500"/>
</p>

<p align="center">
    &nbsp;&nbsp;🤗 <a href="https://huggingface.co/tencnet/TCAndon-Router">Hugging Face</a>&nbsp;&nbsp; | &nbsp;&nbsp; 📑 <a href="https://arxiv.org/pdf/2601.04544">Paper</a> &nbsp;&nbsp;
</p>

\[ English | [中文](README_zh.md) \]

## 🌟 Introduction

In multi-agent systems, the ability to select the appropriate agent(s) to handle a user query is a key determinant of overall system performance.

TCAndonRouter is a reasoning-centric multi-intent routing module whose primary role is to perform agent routing in multi-agent systems.
Beyond agent routing, TCAndonRouter can be applied to any intent-routing scenario, including agent skill selection.

The main advantages of TCAndonRouter include:

+ Designed specifically for real-world enterprise applications
+ Supports dynamic onboarding of new agents (intents) New agents can be added simply by appending their descriptions, without retraining
+ Provides transparent and interpretable routing decisions, improving explainability, robustness, and cross-domain generalization, and making post-deployment bad-case analysis easier
+ Effectively resolves agent conflicts caused by overlapping responsibilities, leading to higher-quality final responses. When multiple agents are applicable, TCAndonRouter preserves all relevant agents. Each downstream agent generates its own response, and a Refining Agent subsequently merges these outputs into a single final answer

TCAndonRouter is trained using Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (DAPO), and achieves state-of-the-art performance on large-scale, real-world enterprise datasets, including HWU64, MINDS14, SGD, and the Tencent Cloud ITSM dataset(QCloud).

| **Models**             | **CLINC150** | **HWU64** | **MINDS14** | **SGD**   | **QCloud**      |
|------------------------|--------------|-----------|-------------|-----------|-----------------|
| GPT-5.1                | 93.84        | 85.59     | 95.59       | 73.90     | 92.80/93.06     |
| Claude-Sonnet-4.5      | **94.21**    | 87.40     | 96.20       | 76.02     | 88.82/94.25     |
| DeepSeek-v3.1-terminus | 88.29        | 88.10     | 95.72       | 79.70     | 94.09/91.89     |
| ArcRouter              | 62.98        | 69.33     | 91.79       | 65.59     | -               |
| Qwen3-Embedding-4B     | 57.21        | 54.27     | 94.12       | 37.02     | -               |
| Qwen3-4B-Instruct-2507 | 70.12        | 80.29     | 90.08       | 58.74     | 82.23/79.44     |
| **TCAndonRouter**           | 91.25        | **91.63** | **96.70**   | **91.58** | **95.21/92.78** |

## 🔧 How to use

### Deploy with vLLM
Configure the model in config/llm_config.toml.
TCAndonRouter supports any LLM, and we recommend using vLLM for deployment when evaluating or serving the router.

See example.py:
```python
from llm_client import completion
from prompt import router_prompt
from utils import load_config

agents = load_config('config/hwu64_config.xml')
query = "Can you recommend any pub in mg road"

prompt = router_prompt.format(agents=agents) + 'user:' + query
message = [{'role': 'user', 'content': prompt}]
response = completion(message)
print(response)
```

### Use HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import router_prompt
from utils import load_config

tokenizer = AutoTokenizer.from_pretrained("tencent/TCAndon-Router")
model = AutoModelForCausalLM.from_pretrained("tencent/TCAndon-Router", device_map="auto")

agents = load_config('config/hwu64_config.xml')
query = "Can you recommend any pub in mg road"
prompt = router_prompt.format(agents=agents) + 'user:' + query

messages = [{"role": "user", "content": prompt}]
encoding = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)

outputs = model.generate(encoding.to(model.device), max_new_tokens=2048)
output_text = tokenizer.decode(outputs[0])
print(output_text)
```

### Evaluate

We provide an evaluation script for four open-source benchmark datasets.

```shell
bash run_eval.sh
```

### Generate Agent Descriptions

If you want to use TCAndonRouter on your own dataset, you need to provide agent descriptions. The required format is defined in `config/xxx_config.xml`.
You can generate agent descriptions using an LLM via generate_agent_desc.py, or write them manually.
```shell
python generate_agent_desc.py --dataset hwu64 --limit 50
```

### Refining Agent
The Refining Agent merges outputs from multiple agents into a single final response. For a complete example, see example.py.

```python
from llm_client import completion
from prompt import refine_prompt

answer_str = ''
for intent in intent_list:
    print('executing intent: ', intent)
    agent_result = agent(query, intent)
    answer_str += f'# The answer of {intent}\n{agent_result}\n'
prompt = refine_prompt.format(query=query, answer=answer_str)
message = [{'role': 'user', 'content': prompt}]
response = completion(message)
```


## 🤝 Citation

If you use TCAndonRouter in your work, please cite our paper:

```
@article{zhao2026TCAndonRouter,
  title={TCAndonRouter: Adaptive Reasoning Router for Multi-Agent Collaboration},
  author={Jiuzhou Zhao, Chunrong Chen, Chenqi Qiao, Lebin Zheng, Minqi Han, Yanchi Liu, Yongzhou Xu, Xiaochuan Xu, Min Zhang},
  journal={arXiv preprint:2601.04544},
  year={2026}
}
```