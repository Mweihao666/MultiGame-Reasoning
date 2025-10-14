# NashEnv evaluation script using vLLM-served model
# Reference: tictactoe_ref.py (mentor's version)
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

def reformat_prompt(prompt0):
    # Append fixed instruction suffix (no chat template, plain concatenation)
    # Note: Avoid mentioning "Nash equilibrium" to test implicit reasoning
    # Balanced version: clear format + strategic hint
    prompt = prompt0 + (
        "\n\nConsider what rational players would do. "
        "Let's think step by step and always output: "
        "<think> [your reasoning] </think> <answer> [1 or 2] </answer> "
        "with no extra text. Strictly follow this format."
    )
    return prompt

def extract_action(output):
    """Extract action from <answer>...</answer> tags"""
    # First try to match the full pattern
    pattern = r'<answer>\s*([12])\s*</answer>'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: find any <answer> tag and extract first 1 or 2
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, output, re.DOTALL)
    if match:
        answer_content = match.group(1)
        # Find first occurrence of 1 or 2
        digit_match = re.search(r'[12]', answer_content)
        if digit_match:
            return digit_match.group(0)
    
    return None

if __name__ == '__main__':
    env = NashEnv()
    info_list = []
    run_details = []  # For saving detailed run info
    
    for t in trange(test_round):
        seed = random.randint(1, 10000)
        prompt = env.reset(seed=seed)
        
        # Format prompt with instruction suffix
        formatted_prompt = reformat_prompt(prompt)
        
        # Query vLLM via OpenAI completions API
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
            run_details.append({
                'episode': t,
                'seed': seed,
                'error': str(e),
                'status': 'api-error'
            })
            print('api-error')
            continue
        
        # Extract action from output
        action_str = extract_action(output)
        
        if action_str is None:
            info_list.append('invalid-format')
            run_details.append({
                'episode': t,
                'seed': seed,
                'output': output,
                'status': 'invalid-format'
            })
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
        
        # Save detailed info (get role and NE from env object)
        NE = env._pure_nash_equilibria() if hasattr(env, '_pure_nash_equilibria') else []
        run_details.append({
            'episode': t,
            'seed': seed,
            'role': env.role if hasattr(env, 'role') else 'unknown',
            'action': action,
            'NE': NE,
            'reward': reward,
            'status': status,
            'output': output[:200]  # Truncate for storage
        })
    
    # Statistics
    counter = Counter(info_list)
    total = len(info_list)
    
    # Analyze by game type
    game_stats = {}
    role_stats = {'P1': {'success': 0, 'total': 0}, 'P2': {'success': 0, 'total': 0}}
    for detail in run_details:
        if 'NE' not in detail or detail.get('status') == 'invalid-format':
            continue
        
        # Infer game type from NE
        ne = detail['NE']
        if len(ne) == 1:
            game_type = 'PD'  # 囚徒困境：唯一纳什均衡
        elif len(ne) == 2:
            game_type = 'SH'  # 猎鹿博弈：两个纳什均衡
        elif len(ne) == 0:
            game_type = 'MP'  # 配对硬币：无纯策略纳什均衡
        else:
            game_type = 'Other'
        
        if game_type not in game_stats:
            game_stats[game_type] = {'total': 0, 'success': 0}
        game_stats[game_type]['total'] += 1
        if detail['status'] == 'success':
            game_stats[game_type]['success'] += 1
        
        # Role stats
        role = detail.get('role', 'Unknown')
        if role in role_stats:
            role_stats[role]['total'] += 1
            if detail['status'] == 'success':
                role_stats[role]['success'] += 1
    
    # Print summary
    print("\n" + "="*50)
    print(f"NashEnv Test Results (n={total})")
    print("="*50)
    for key, value in counter.items():
        print(f"{key}: {value} ({value / total:.2%})")
    
    success_count = counter.get('success', 0)
    print(f"\nSuccess rate: {success_count / total:.2%}")
    
    # Print by game type
    print("\n【By Game Type】")
    for game_type in ['PD', 'SH', 'MP', 'Other']:
        if game_type in game_stats:
            stats = game_stats[game_type]
            rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {game_type}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
    
    # Print by role
    print("\n【By Role】")
    for role in ['P1', 'P2']:
        stats = role_stats[role]
        rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {role}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
    
    print("="*50)
    
    # Save results to log file
    with open('reason_test/nashenv-log.txt', 'a') as f:
        f.write(f"\n=== Model: {args.model_path} ===\n")
        f.write(f"Test date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test rounds: {total}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write("NashEnv test results:\n")
        for key, value in counter.items():
            f.write(f"  {key}: {value} ({value / total:.2%})\n")
        f.write(f"Success rate: {success_count / total:.2%}\n")
        
        # Save game type breakdown
        f.write("\nBy Game Type:\n")
        for game_type in ['PD', 'SH', 'MP']:
            if game_type in game_stats:
                stats = game_stats[game_type]
                rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
                f.write(f"  {game_type}: {stats['success']}/{stats['total']} ({rate:.1f}%)\n")
        
        # Save role breakdown
        f.write("\nBy Role:\n")
        for role in ['P1', 'P2']:
            stats = role_stats[role]
            rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
            f.write(f"  {role}: {stats['success']}/{stats['total']} ({rate:.1f}%)\n")
        
        f.write(f"\nPort: {port}\n\n")
    
    # Save detailed run info
    with open('reason_test/nashenv-last-run.jsonl', 'w') as f:
        for detail in run_details:
            f.write(json.dumps(detail, ensure_ascii=False) + '\n')
    
    print(f"\nResults saved to reason_test/nashenv-log.txt")
    print(f"Detailed run saved to reason_test/nashenv-last-run.jsonl")
