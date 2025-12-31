# 改动之前的tictactoe测试代码，改为使用python的VLLM load训练模型、并尝试初始化对手的config（是否有可能）
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragen.env.tictactoe.config import TicTacToeEnvConfig
from ragen.env.tictactoe.env import TicTacToeEnv
# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import time
from tqdm import trange
import random
import os
from collections import Counter
import argparse
from ragen.env.base import EnvPlayer

root_path = '/root/autodl-tmp'  # '/data1/lvnuoyan' 
test_round = 100
config = TicTacToeEnvConfig(
    max_env_try=1,  # 修改最大尝试次数
    player_info=[
        {'model_name': 'google/gemini-2.5-flash-nothinking'}
    ]
)
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='google/gemini-2.5-flash-nothinking', 
                    help="模型名称，支持 gemini (google/gemini-2.5-flash-nothinking) 或 gpt-4o")
parser.add_argument("--max_rounds", type=int, default=None, help="最大测试轮数（None表示使用默认轮数）")
parser.add_argument("--max_tokens", type=int, default=3000, help="最大生成 token 数")
args = parser.parse_args()
model_name = args.model_name

# 创建 EnvPlayer（用于 API 调用）
env_player = EnvPlayer(2, {'model_name': model_name}, max_tokens=args.max_tokens)

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

def load_llm():
    """兼容性函数，返回 env_player 和 None"""
    return env_player, None


def reformat_prompt(prompt0):
    # 将prompt句子末尾的 Let\'s think step by step and output the final answer after "####".
    # 替换为Let\'s think step by step and output your think and final answer in this format: 
    # <think> [your thought] </think> <answer> [your answer] </answer>
    prompt = prompt0 + "Let\'s think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens)."
    # 注意：chat template 现在由 model_adapter 处理
    return prompt

if __name__ == '__main__':
    llm, sampling_params = load_llm()
    env = TicTacToeEnv(config=config)
    info_list = []
    actual_rounds = args.max_rounds if args.max_rounds else test_round
    for t in trange(actual_rounds):
        env.reset(seed=random.randint(1, 1000))
        # 游戏进行
        prompt = env.render()
        turn = 0
        while True:
            turn += 1
            # 打印输入（用于检查）
            print(f"\n=== Round {t+1}, Turn {turn} ===")
            print(f"Input (env.render()):\n{prompt}\n")
            # 得到trainer的行动
            formatted_prompt = reformat_prompt(prompt)
            print(f"Formatted Input (with format prompt):\n{formatted_prompt[:500]}...\n")
            # 使用 EnvPlayer 生成（llm_output 内部已处理重试）
            output = llm_output(formatted_prompt)
            # 打印输出（用于检查）
            print(f"Output (API response):\n{output}\n")
            # 去掉输出末尾的空白字符，避免匹配失败
            output_stripped = output.strip()
            # 首先尝试完整格式匹配
            pattern = r'.*<think>(.*?)</think>\s*<answer>(.*?)</answer>$'
            match = re.match(pattern, output_stripped, re.DOTALL)
            if not match:
                # 如果完整格式匹配失败，尝试只匹配answer标签（处理输出被截断的情况）
                answer_pattern = r'<answer>(.*?)</answer>'
                answer_match = re.search(answer_pattern, output_stripped, re.DOTALL)
                if answer_match:
                    action = answer_match.group(1).strip()
                    print(f"  警告: 输出格式不完整（缺少think标签），但已提取答案: {action}")
                else:
                    info_list.append('trainer-invalid-format')
                    print('trainer-invalid-format')
                    break
            else:
                action = match.group(2).strip()
            # 更新环境信息，得到对手操作以及下一步信息
            prompt, reward, done, info = env.step(action)
            # 检查游戏是否结束，更改了invalid——这里设置invalid为游戏失败、直接结束
            if "Invalid action:" in prompt:
                info_list.append('trainer-invalid-output')
                print('trainer-invalid-output')
                break
            if done:
                # 如果结束检查游戏状态，添加对应的状态信息
                if "Congratulations!" in prompt:
                    info_list.append('success')
                    print('success')
                elif "Draw!" in prompt:
                    info_list.append('draw')
                    print('draw')
                # invalid: env_player不符合指令遵循
                # wrong：env_player的动作不在available_action中
                elif "Your opponent made a mistake" in prompt:
                    info_list.append('env_player-invalid-output')
                    print('env_player-invalid-output')
                elif "Your opponent action is wrong" in prompt:
                    info_list.append('env_player-wrong-output')
                    print('env_player-wrong-output')
                elif "Failed! " in prompt:
                    info_list.append('fail')
                    print('fail')
                break
    # 统计info_list中出现的不同情况的次数并计算对应的比例
    counter = Counter(info_list)
    total = len(info_list)
    # 存储结果文件
    with open('reason_test/tictactoe-log.txt', 'a') as f:
        f.write(f"\n=== Model: {model_name} ===\n")
        f.write("tictactoe test set\n")
        for key, value in counter.items():
            f.write(f"{key}: {value / total:.2%}\n")
        f.write(f"tictactoe v.s. {config.player_info[0]['model_name']}\n")
        f.write(f"model: {model_name}\n\n")
 

