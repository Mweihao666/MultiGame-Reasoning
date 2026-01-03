# NashEnv evaluation script using vLLM-served model
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragen.env.nash_new.env import NashNew
from ragen.env.nash_new.config import NashNewConfig
import json
import re
import time
from tqdm import trange
import random
from collections import Counter
import argparse
from ragen.env.base import EnvPlayer

# Setup
root_path = '/root/autodl-tmp'
test_round = 100

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='google/gemini-2.5-flash-nothinking', 
                    help="模型名称，支持 gemini (google/gemini-2.5-flash-nothinking) 或 gpt-4o")
parser.add_argument("--max_rounds", type=int, default=None, help="最大测试轮数（None表示使用默认轮数）")
parser.add_argument("--max_tokens", type=int, default=3000, help="最大生成 token 数")
args = parser.parse_args()

model_name = args.model_name
# 如果模型名称是 gpt-4o，转换为 openai/gpt-4o（通过 openrouter 调用）
if model_name == 'gpt-4o':
    model_name = 'openai/gpt-4o'

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
                print(f" API 返回空结果，{wait_time} 秒后重试 (第 {retries} 次)...")
                time.sleep(wait_time)
        except Exception as e:
            retries += 1
            wait_time = min(2 ** retries, 30)  # 指数退避，最多等待30秒
            print(f"  API 调用失败: {e}，{wait_time} 秒后重试 (第 {retries} 次)...")
            time.sleep(wait_time)

def load_llm():
    """兼容性函数，返回 env_player 和 None"""
    return env_player, None

def reformat_prompt(prompt0):
    # Append fixed instruction suffix (no chat template, plain concatenation)
    # Guide model to analyze payoffs without mentioning "Nash equilibrium"
    prompt = prompt0 + "Let\'s think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens)."
    # 注意：chat template 现在由 model_adapter 处理
    return prompt

def extract_action(output):
    """Extract action from <answer>...</answer> tags"""
    # First try to match the full pattern
    pattern = r'<answer>\s*(.*)\s*</answer>'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # # Fallback: find any <answer> tag and extract first 1 or 2
    # answer_pattern = r'<answer>(.*?)</answer>'
    # match = re.search(answer_pattern, output, re.DOTALL)
    # if match:
    #     answer_content = match.group(1)
    #     # Find first occurrence of 1 or 2
    #     digit_match = re.search(r'[12]', answer_content)
    #     if digit_match:
    #         return digit_match.group(0)
    
    return None

if __name__ == '__main__':
    env = NashNew()
    info_list = []
    run_details = []  # For saving detailed run info
    llm, sampling_params = load_llm()
    actual_rounds = args.max_rounds if args.max_rounds else test_round
    for t in trange(actual_rounds):
        seed = random.randint(1, 1000)
        prompt = env.reset(seed=seed)
        # Format prompt with instruction suffix
        formatted_prompt = reformat_prompt(prompt)
        # 打印输入（用于检查）
        print(f"\n=== Round {t+1} ===")
        print(f"Input (env.render()):\n{prompt}\n")
        print(f"Formatted Input (with format prompt):\n{formatted_prompt[:500]}...\n")
        # 使用 EnvPlayer 生成（llm_output 内部已处理重试）
        output = llm_output(formatted_prompt)
        # 打印输出（用于检查）
        print(f"Output (API response):\n{output}\n")
        print(f"Output length: {len(output)} characters\n")
        # 检查是否包含结束标签
        if '</think>' not in output:
            print("  警告: 输出中缺少 </think> 标签")
        if '</answer>' not in output:
            print("  警告: 输出中缺少 </answer> 标签")
        # Extract action from output
        action_str = extract_action(output)
        if action_str is None:
            info_list.append('invalid-format')
            print('invalid-format')
            continue
        # Keep action as string for env.step() (导师版本需要字符串)
        action = action_str
        # Step environment
        prompt, reward, done, info = env.step(action)
        # Record result
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
    
    # Statistics
    counter = Counter(info_list)
    total = len(info_list)
    
    # Print summary
    print("\n" + "="*20)
    print(f"NashEnv Test Results (n={total})")
    print("="*20)
    for key, value in counter.items():
        print(f"{key}: {value / total:.2%}")
    
    # success_count = counter.get('success', 0)
    # print(f"\nSuccess rate: {success_count / total:.2%}")
    print("="*20)
    
    # Save results to log file
    with open('reason_test/nashenv-log.txt', 'a') as f:
        f.write(f"\n=== Model: {args.model_name} ===\n")
        f.write("NashEnv test results:\n")
        for key, value in counter.items():
            f.write(f"{key}: {value / total:.2%}\n")
        # f.write(f"Success rate: {success_count / total:.2%}\n")
    print(f"\nResults saved to reason_test/nashenv-log.txt")