# TCAndon-Router

<p align="center">
    <img src="assets/router.png" width="500"/>
</p>

<p align="center">
    &nbsp;&nbsp;🤗 <a href="https://huggingface.co/tencent/TCAndon-Router">Hugging Face</a>&nbsp;&nbsp; | &nbsp;&nbsp; 📑 <a href="https://arxiv.org/pdf/2601.04544">Paper</a> &nbsp;&nbsp;
</p>

\[ [English](README.md) | 中文 \]

## 🌟 简介
在多agent系统中，能否选择合适的agent来解决用户的问题决定了整个系统的效果。

TCAndonRouter 是一个以推理为核心的多意图路由模块，其主要任务是多agent系统中的agent路由选择，除此之外，你也可以把TCAndonRouter用在任何的意图场景包括agent skills的选择，TCAndonRouter的优势为：

+ 专为真实企业应用场景设计
+ 支持新agent(意图)的动态接入，添加新agent只需追加新的agent描述，无需重新训练
+ 提供透明且可解释的路由决策，提升了路由的可解释性、鲁棒性和跨领域泛化能力，业务上线后便于解决badcase
+ 有效解决因职责重叠导致的agent冲突，生成更高质量的最终答案，当多个agent都适用时，TCAndonRouter 会保留所有相关agent，下游agent各自生成响应，Refining Agent随后将这些输出合并为单一的最终回复

TCAndonRouter 采用监督微调（SFT）+ 强化学习（DAPO）进行训练，在 HWU64、MINDS14、SGD 以及腾讯云ITSM数据集(QCloud)等大规模真实企业数据集上达到了SOTA的效果。

| **模型**               | **CLINC150** | **HWU64** | **MINDS14** | **SGD**   | **QCloud**      |
|------------------------|--------------|-----------|-------------|-----------|-----------------|
| GPT-5.1                | 93.84        | 85.59     | 95.59       | 73.90     | 92.80/93.06     |
| Claude-Sonnet-4.5      | **94.21**    | 87.40     | 96.20       | 76.02     | 88.82/94.25     |
| DeepSeek-v3.1-terminus | 88.29        | 88.10     | 95.72       | 79.70     | 94.09/91.89     |
| ArcRouter              | 62.98        | 69.33     | 91.79       | 65.59     | -               |
| Qwen3-Embedding-4B     | 57.21        | 54.27     | 94.12       | 37.02     | -               |
| Qwen3-4B-Instruct-2507 | 70.12        | 80.29     | 90.08       | 58.74     | 82.23/79.44     |
| **TCAndonRouter**           | 91.25        | **91.63** | **96.70**   | **91.58** | **95.21/92.78** |

## 🔧 使用方法

### 使用vllm

修改 `config/llm_config.toml` 支持任意的llm，如需测试TCAndonRouter，建议使用vllm进行部署并修改该配置。
参考 `example.py`：
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

### 使用 HuggingFace

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

### 评估

我们提供了一个脚本来评估四个开源数据集。
```shell
bash run_eval.sh
```

### 生成agent描述

如果你想在自己的数据集上使用 TCAndonRouter，需要提供agent描述，所需格式定义在 `config/xxx_config.xml` 中。
你可以使用 `generate_agent_desc.py` 通过 LLM 生成描述，或手动编写。
```shell
python generate_agent_desc.py --dataset hwu64 --limit 50
```

### Refining Agent
Refining Agent将多个agent的输出合并为单一的最终回复，更多详情请参考 `example.py`。

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


## 🤝 引用

如果你在工作中使用了TCAndonRouter，请引用我们的paper：

```
@article{zhao2026TCAndonRouter,
  title={TCAndonRouter: Adaptive Reasoning Router for Multi-Agent Collaboration},
  author={Jiuzhou Zhao, Chunrong Chen, Chenqi Qiao, Lebin Zheng, Minqi Han, Yanchi Liu, Yongzhou Xu, Xiaochuan Xu, Min Zhang},
  journal={arXiv preprint:2601.04544},
  year={2026}
}
```

