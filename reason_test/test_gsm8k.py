import datasets
import re
import tqdm
import json
import argparse
import os
import time
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragen.env.base import EnvPlayer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='google/gemini-2.5-flash-nothinking', 
                    help="模型名称，支持 gemini (google/gemini-2.5-flash-nothinking) 或 gpt-4o")
parser.add_argument('--max_samples', type=int, default=None, help="最大测试样本数（None表示使用全部样本）")
parser.add_argument('--max_tokens', type=int, default=1000, help="最大生成 token 数")
args = parser.parse_args()

model_name = args.model_name
# 如果模型名称是 gpt-4o，转换为 openai/gpt-4o（通过 openrouter 调用）
if model_name == 'gpt-4o':
    model_name = 'openai/gpt-4o'
root_path = '/root/autodl-tmp'

# 创建 EnvPlayer（用于 API 调用）
env_player = EnvPlayer(2, [{'model_name': model_name}], max_tokens=args.max_tokens)

def llm_output(text: str) -> str:
    """通过 EnvPlayer 调用模型获取输出，失败时重试直到成功"""
    retries = 0
    while True:
        try:
            output = env_player.act(text, 0)
            if output:  # 如果返回非空字符串，认为成功
                return output
            else:
                retries += 1
                wait_time = min(2 ** retries, 30)  # 指数退避，最多等待30秒
                print(f"  API 返回空结果，{wait_time} 秒后重试 (第 {retries} 次)...")
                time.sleep(wait_time)
        except Exception as e:
            retries += 1
            wait_time = min(2 ** retries, 30)  # 指数退避，最多等待30秒
            print(f"  API 调用失败: {e}，{wait_time} 秒后重试 (第 {retries} 次)...")
            time.sleep(wait_time)

def reformat_prompt(prompt0):
    # 将prompt句子末尾的 Let\'s think step by step and output the final answer after "####".
    # 替换为Let\'s think step by step and output your think and final answer in this format: 
    # <think> [your thought] </think> <answer> [your answer] </answer>
    prompt = prompt0.replace("Let\'s think step by step and output the final answer after \"####\".", 
                             "Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens).")
    # 注意：对于 API 模型（deepseek, gemini），prompt 直接作为 user message
    # 对于本地模型（vllm, bbl-lite），model_adapter 会应用 chat template
    return prompt

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if method == "strict":
        # this also tests the formatting of the model
        # 更精确的正则表达式
        pattern = r"<answer>.*?(-?\d+\.?\d*)[^0-9]*</answer>"
        # 使用 re.search() 提取最后一个数字
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            last_number = match.group(1)  # 提取匹配到的最后一个数字
            return last_number
        else:
            return None 
    elif method == "flexible":
        # 但是这个flexible匹配明明已经是宽松版本了，但还是匹配不到——可能还是训废了？        
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def test_math(method='strict', max_samples=None):
    accs = []
    answers = []
    num_samples = min(len(math['test']), max_samples) if max_samples else len(math['test'])
    for i in tqdm.trange(num_samples):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲
        q = math['test']['prompt'][i][0]['content']
        q = reformat_prompt(q)
        # print(q)
        ground_truth = math['test']['reward_model'][i]['ground_truth']
        a = llm_output(q)
        # print(a)
        answers.append(a)
        solution = extract_solution(a, method)
        # print(solution)
        # print(type(ground_truth))
        # 之前发现可能表达式不同但实际上是一个数值的情况，比如100.00和100，然后可能有多余的
        # 先不考虑这个情况
        if solution is None:
            accs.append(None)
        elif solution == ground_truth:
            accs.append(1)
        else:
            accs.append(0)
    return accs, answers


path0 = f'{root_path}/reasoning'
math = datasets.load_dataset("parquet", 
              data_files={'train': path0 + '/gsm8k/train.parquet', 'test': path0 + '/gsm8k/test.parquet'})
# print(math['test']['prompt'][0])

# exit(0)
accs, answers = test_math('strict', max_samples=args.max_samples)
acc0 = accs.count(1) / len(accs)
print('total acc:', acc0)
print('invalid output:', accs.count(None))
TIME = datetime.now().strftime("%m-%d-%H-%M")
# 删除特殊字符
model_name = model_name.replace('/', '').replace('\\', '')
with open(f'{model_name}-gsm8k-{TIME}.json', 'w') as f:
    f.write(json.dumps(answers))
