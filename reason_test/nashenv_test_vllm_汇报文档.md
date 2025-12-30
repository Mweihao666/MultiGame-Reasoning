# NashEnv 测试脚本开发汇报

**汇报人：** [你的名字]  
**时间：** 2025年10月14日  
**脚本：** `reason_test/nashenv_test_vllm.py`

---

## 一、脚本整体结构

这个测试脚本总共 180 行，主要分为几个部分：
1. 导入和配置（1-36行）
2. Prompt 格式化（38-48行）
3. 动作提取（50-67行）
4. 主测试循环（69-145行）
5. 统计和保存（147-180行）

下面我逐段讲解每部分的作用。

---

## 二、核心函数解析

### **【第 1-36 行】导入和配置**

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nashenv import NashEnv, NashEnvConfig
import json
import re
import time
from tqdm import trange
import random
from collections import Counter
from openai import OpenAI
import argparse

# Setup
root_path = '/root/autodl-tmp'
test_round = 50

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen2.5-1.5B-Instruct")
parser.add_argument("--port", type=int, default=2515)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--test_round", type=int, default=50)
args = parser.parse_args()

model_path = f"{root_path}/{args.model_path}"
port = args.port
temperature = args.temperature
test_round = args.test_round

# Initialize OpenAI client for vLLM
client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://0.0.0.0:{port}/v1"
)
```

这部分主要是导入必要的库和配置参数。我用 `argparse` 让脚本支持命令行参数，可以灵活调整测试轮数、温度、端口等。然后初始化 OpenAI 客户端，连接到本地的 vLLM 服务（端口 2515）。这个客户端是兼容 OpenAI API 的，所以用起来很方便。

---

### **【第 38-48 行】Prompt 格式化**

```python
def reformat_prompt(prompt0):
    # Append fixed instruction suffix (no chat template, plain concatenation)
    # Guide model to analyze payoffs without mentioning "Nash equilibrium"
    prompt = prompt0 + (
        "\n\nAnalyze the payoff matrix carefully. Consider what each player would choose "
        "to maximize their own payoff, and select your best action accordingly.\n\n"
        "Let's think step by step and always output: "
        "<think> [Your thoughts] </think> <answer> [your action: 1 or 2] </answer> "
        "with no extra text. Strictly follow this format."
    )
    return prompt
```

这个函数负责把环境给的博弈 prompt 和我们的指令拼在一起。这里有个重要的设计点：**我让模型分析收益矩阵、考虑每个玩家会怎么选，但是绝对不提"纳什均衡"这四个字**。这样就能测试模型是否真的能从收益矩阵推理出理性策略，而不是靠记住博弈论概念。

然后要求模型按固定格式输出，方便后面提取答案。

---

### **【第 50-67 行】动作提取**

```python
def extract_action(output):
    # 先尝试精确匹配
    pattern = r'<answer>\s*([12])\s*</answer>'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 如果不行，就宽松一点，只要在 <answer> 标签里找到 1 或 2 就行
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, output, re.DOTALL)
    if match:
        answer_content = match.group(1)
        digit_match = re.search(r'[12]', answer_content)
        if digit_match:
            return digit_match.group(0)
    
    return None  # 实在找不到就返回 None
```

这个函数从模型的输出里提取动作。我用了两层策略：先试精确匹配，匹配不上就放宽条件。因为有时候模型输出格式不太标准，但只要在 `<answer>` 标签里能找到 1 或 2，我们就认为是有效的。如果实在提取不出来，就返回 `None`，后面会标记为格式错误。

---

## 三、主测试流程

### **【第 69-80 行】循环初始化**

```python
if __name__ == '__main__':
    env = NashEnv()
    info_list = []    # 记录每轮结果：success/fail/invalid-format
    run_details = []  # 记录详细信息
    
    for t in trange(test_round):
        seed = random.randint(1, 10000)
        prompt = env.reset(seed=seed)
```

这里就是主流程开始。我创建了环境，准备两个列表分别记录简要结果和详细信息。然后开始循环，每轮用随机种子重置环境，拿到新的博弈 prompt。`trange` 会显示进度条，看起来比较直观。

---

### **【第 82-96 行】调用模型**

```python
        formatted_prompt = reformat_prompt(prompt)
        
        try:
            response = client.completions.create(
                model=model_path,
                prompt=formatted_prompt,
                max_tokens=600,
                temperature=temperature
            )
            output = response.choices[0].text
        except Exception as e:
            info_list.append('api-error')
            run_details.append({...})
            print('api-error')
            continue
```

拿到 prompt 后，先拼接指令，然后发给 vLLM。我用的是 `completions` 接口，因为是纯文本拼接，不需要对话格式。如果调用失败了，就记录 API 错误然后继续下一轮，不让整个测试中断。

---

### **【第 98-109 行】提取动作**

```python
        action_str = extract_action(output)
        
        if action_str is None:
            info_list.append('invalid-format')
            run_details.append({...})
            print('invalid-format')
            continue
        
        action = action_str  # 保持字符串类型
```

用刚才那个函数提取动作。如果提取失败，说明模型输出格式有问题，记为 `invalid-format` 然后跳过。这里有个小细节，**动作要保持字符串类型**，因为环境的接口需要的是字符串 `'1'` 或 `'2'`，不是整数。

---

### **【第 111-133 行】执行和判断**

```python
        prompt, reward, done, info = env.step(action)
        
        if info.get('success', False):
            info_list.append('success')
            status = 'success'
            print('success')
        else:
            if not info.get('action_is_valid', True):
                info_list.append('invalid-action')
                status = 'invalid-action'
                print('invalid-action')
            else:
                info_list.append('fail')
                status = 'fail'
                print('fail')
```

把动作传给环境，环境会返回奖励和信息。然后根据 `info['success']` 判断这轮是成功还是失败。我会实时打印每轮结果，这个是参考您 TicTacToe 脚本的风格。成功就打 `success`，失败就打 `fail`。

---

### **【第 134-145 行】保存详细信息**

```python
        NE = env._pure_nash_equilibria() if hasattr(env, '_pure_nash_equilibria') else []
        run_details.append({
            'episode': t,
            'seed': seed,
            'role': env.role if hasattr(env, 'role') else 'unknown',
            'action': action,
            'NE': NE,
            'reward': reward,
            'status': status,
            'output': output[:200]
        })
```

每轮结束后，我会保存详细信息。包括种子、角色（P1还是P2）、选择的动作、纳什均衡是什么、以及模型的输出（截断到200字符）。这些信息后面可以用来做详细分析。

---

## 四、统计和保存

### **【第 147-160 行】统计打印**

```python
    counter = Counter(info_list)
    total = len(info_list)
    
    print("\n" + "="*50)
    print(f"NashEnv Test Results (n={total})")
    print("="*50)
    for key, value in counter.items():
        print(f"{key}: {value} ({value / total:.2%})")
    
    success_count = counter.get('success', 0)
    print(f"\nSuccess rate: {success_count / total:.2%}")
    print("="*50)
```

所有轮次跑完后，用 `Counter` 统计一下各种结果的数量。然后打印出来，包括成功多少次、失败多少次、格式错误多少次，以及总成功率。输出格式也是参考 TicTacToe 的风格。

---

### **【第 162-180 行】保存日志**

```python
    # 保存汇总日志
    with open('reason_test/nashenv-log.txt', 'a') as f:
        f.write(f"\n=== Model: {args.model_path} ===\n")
        f.write(f"Test date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        # ... 写入统计信息
    
    # 保存详细记录
    with open('reason_test/nashenv-last-run.jsonl', 'w') as f:
        for detail in run_details:
            f.write(json.dumps(detail, ensure_ascii=False) + '\n')
```

最后保存两个文件。一个是 `nashenv-log.txt`，追加模式保存汇总统计，方便对比不同测试的结果。另一个是 `nashenv-last-run.jsonl`，保存每轮的详细信息，一行一个 JSON 对象，后面如果要分析哪里出问题了就很方便。

---

## 五、测试结果

我测试了 50 轮，用的是 Qwen2.5-1.5B-Instruct，温度设为 0（确定性输出）。结果是：

- **成功率：50%**
- 失败率：46%
- 格式错误：4%

这个 50% 其实是等于理论上的随机猜测基线。说明什么呢？说明这个未经训练的基座模型理解了任务，知道要选 1 或 2，但是它没有足够的博弈论推理能力来找到正确答案。这正好可以作为一个基线，后面如果训练了模型，可以用这个脚本测一下，看能不能提升到 70%、80% 甚至更高。

---

## 六、使用方式

使用很简单。最基础的就是：

```bash
python reason_test/nashenv_test_vllm.py
```

如果要自定义参数，比如：

```bash
python reason_test/nashenv_test_vllm.py --test_round 100 --temperature 0.5
```

就会测 100 轮，温度设为 0.5。

---

**以上就是整个脚本的讲解。导师您看有什么问题吗？**

