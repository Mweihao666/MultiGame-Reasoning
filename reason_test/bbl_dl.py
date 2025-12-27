# BIG-bench Lite (BBL) 任务评测脚本
# 参考 test_gsm8k.py 的思路：通过 API 调用 vLLM 服务
# 支持 logical_deduction 等多项选择任务

from openai import OpenAI
import json
import re
import tqdm
import argparse
import os
import time
from datetime import datetime
from verl.utils import hf_tokenizer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default="2100", help="vLLM 服务端口")
parser.add_argument('--model_path', type=str, default="nash-new", help="模型路径（相对于 root_path）")
parser.add_argument('--model_name', type=str, default="nash50", help="模型名称")
parser.add_argument('--task_name', type=str, default="logical_deduction", help="BBL 任务名称")
parser.add_argument('--task_subdir', type=str, default="five_objects", help="任务子目录（如 five_objects）")
parser.add_argument('--task_file', type=str, default=None, help="直接指定 task.json 文件路径（可选）")
args = parser.parse_args()

port = args.port
model_path = args.model_path
model_name = args.model_name
task_name = args.task_name
task_subdir = args.task_subdir
task_file = args.task_file

root_path = '/root/autodl-tmp'
# 尝试初始化 tokenizer，如果路径不存在则使用备用模型
try:
    if model_path:
        tokenizer_path = f"{root_path}/{model_path}/{model_name}"
    else:
        tokenizer_path = f"{root_path}/{model_name}"
    # 检查路径是否存在
    if os.path.exists(tokenizer_path):
        tokenizer = hf_tokenizer(tokenizer_path)
    else:
        # 如果路径不存在，使用 Qwen2.5-1.5B-Instruct 
        print(f"模型路径不存在 {tokenizer_path}，使用备用 tokenizer: Qwen2.5-1.5B-Instruct")
        tokenizer = hf_tokenizer(f"{root_path}/Qwen2.5-1.5B-Instruct")
except Exception as e:
    # 如果初始化失败，使用备用模型
    print(f"tokenizer 初始化失败: {e}，使用备用 tokenizer: Qwen2.5-1.5B-Instruct")
    tokenizer = hf_tokenizer(f"{root_path}/Qwen2.5-1.5B-Instruct")

# 禁用代理（只在本脚本有效）
for key in ["http_proxy", "https_proxy", "all_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(key, None)

# 配置 OpenAI 客户端（兼容 vLLM 的 OpenAPI 接口）
client = OpenAI(
    api_key="EMPTY",  # vLLM 无需认证密钥，任意字符串均可
    base_url=f"http://localhost:{port}/v1"  # 与 vLLM 服务端口一致
)

def llm_output(text: str) -> str:
    """通过 API 调用模型获取输出（参考 test_gsm8k.py）"""
    try:
        message = [
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": text}
        ]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        
        # 构建模型路径（处理空路径的情况）
        if model_path:
            model_full_path = f"{root_path}/{model_path}/{model_name}"
        else:
            model_full_path = f"{root_path}/{model_name}"
        
        response = client.completions.create(
            model=model_full_path,
            prompt=prompt,
            max_tokens=600,
            temperature=0.5,
        )
        return response.choices[0].text
    except Exception as e:
        raise RuntimeError(f"API 调用失败：{str(e)}")

def load_task_json(task_file_path: str) -> dict:
    """加载 task.json 文件"""
    with open(task_file_path, 'r', encoding='utf-8') as f:
        task_data = json.load(f)
    return task_data

def reformat_prompt(example: dict, task_prefix: str = "") -> str:
    """
    格式化 prompt（参考 test_gsm8k.py）
    将多项选择任务的选项格式化为 A/B/C/D... 的形式
    """
    input_text = example.get('input', '')
    target_scores = example.get('target_scores', {})
    
    # 构建 prompt
    if task_prefix:
        prompt = f"{task_prefix}\n\n{input_text}"
    else:
        prompt = input_text
    
    # 添加选项（格式化为 A/B/C/D...）
    choices = list(target_scores.keys())
    prompt += "\n\nOptions:"
    for i, choice in enumerate(choices):
        prompt += f"\n{chr(65 + i)}. {choice}"
    
    # 添加输出格式要求（参考 test_gsm8k.py）
    prompt += "\n\nLet's think step by step and always output: <think> [Your thoughts] </think> <answer> </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens)."
    
    return prompt

def extract_answer(solution_str: str, num_choices: int, choices_are_numeric: bool = False) -> str:
    """
    从模型输出中提取答案（A/B/C/D... 或 1/2/3/4...）
    """
    # 先尝试从 <answer> 标签中提取
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
    else:
        answer_text = solution_str
    
    if choices_are_numeric:
        # 如果选项是数字，提取数字并转换为字母
        # 提取 1-9 的数字（在合理范围内）
        for i in range(1, min(num_choices + 1, 10)):
            pattern = rf'(?<![0-9]){i}(?![0-9]|\.)'
            if re.search(pattern, answer_text):
                # 转换为字母：1->A, 2->B, ...
                return chr(64 + i)  # 65-1=64, 所以 1->A(65), 2->B(66)
    else:
        # 提取第一个 A-Z 字母（在合理范围内）
        valid_letters = [chr(65 + i) for i in range(min(num_choices, 26))]
        for letter in valid_letters:
            # 匹配独立的字母（允许后面跟点或空格，但不跟其他字母）
            pattern = rf'(?<![A-Za-z]){letter}(?:\s|\.|$|[^A-Za-z])'
            if re.search(pattern, answer_text, re.IGNORECASE):
                return letter.upper()
    
    return None

def test_bbl(task_data: dict):
    """
    评测 BBL 任务（参考 test_gsm8k.py 的 test_math 函数）
    """
    examples = task_data.get('examples', [])
    task_prefix = task_data.get('task_prefix', '')
    
    accs = []
    answers = []
    
    # 获取选项数量（假设所有样本的选项数量相同）
    if examples:
        num_choices = len(examples[0].get('target_scores', {}))
        # 检查选项是否是数字（判断第一个样本的第一个选项是否是纯数字）
        first_choice = list(examples[0].get('target_scores', {}).keys())[0] if examples[0].get('target_scores') else ""
        choices_are_numeric = first_choice.isdigit() if first_choice else False
    else:
        num_choices = 4
        choices_are_numeric = False
    
    for i in tqdm.trange(len(examples)):
        example = examples[i]
        
        # 格式化 prompt
        prompt = reformat_prompt(example, task_prefix)
        
        # 调用 API 获取模型输出
        try:
            output = llm_output(prompt)
            answers.append(output)
        except Exception as e:
            print(f"⚠️  样本 {i} API 调用失败: {e}")
            answers.append("")
            accs.append(None)
            continue
        
        # 提取答案（支持数字选项）
        extracted_choice = extract_answer(output, num_choices, choices_are_numeric)
        
        # 获取正确答案
        target_scores = example.get('target_scores', {})
        correct_choices = [choice for choice, score in target_scores.items() if score == 1]
        
        if not correct_choices:
            # 如果没有找到得分为1的选项，尝试找得分最高的
            max_score = max(target_scores.values()) if target_scores else 0
            correct_choices = [choice for choice, score in target_scores.items() if score == max_score]
        
        if correct_choices:
            # 将正确答案映射到字母
            all_choices = list(target_scores.keys())
            correct_letter = chr(65 + all_choices.index(correct_choices[0]))
            
            # 比较答案
            if extracted_choice == correct_letter:
                accs.append(1)
            else:
                accs.append(0)
        else:
            accs.append(None)
    
    return accs, answers

# 主程序
if __name__ == '__main__':
    # 确定 task.json 文件路径
    if task_file:
        task_json_path = task_file
    else:
        # 默认路径：假设 BIG-bench 仓库在 /root/autodl-tmp/BIG-bench
        if task_subdir:
            task_json_path = f"/root/autodl-tmp/BIG-bench/bigbench/benchmark_tasks/{task_name}/{task_subdir}/task.json"
        else:
            task_json_path = f"/root/autodl-tmp/BIG-bench/bigbench/benchmark_tasks/{task_name}/task.json"
    
    print(f"正在加载任务文件: {task_json_path}")
    
    if not os.path.exists(task_json_path):
        print(f"❌ 错误: 任务文件不存在: {task_json_path}")
        print("\n请确保:")
        print("1. BIG-bench 仓库已克隆到 /root/autodl-tmp/BIG-bench")
        print("2. 或使用 --task_file 参数指定 task.json 文件的完整路径")
        exit(1)
    
    # 加载任务数据
    task_data = load_task_json(task_json_path)
    print(f"任务名称: {task_data.get('name', 'Unknown')}")
    print(f"任务描述: {task_data.get('description', 'N/A')[:100]}...")
    print(f"测试样本数: {len(task_data.get('examples', []))}")
    print(f"vLLM 服务端口: {port}")
    print(f"模型路径: {root_path}/{model_path}/{model_name}")
    print("\n开始评测...")
    
    # 运行评测
    accs, answers = test_bbl(task_data)
    
    # 计算准确率
    total = len(accs)
    correct = accs.count(1)
    invalid = accs.count(None)
    acc0 = correct / total if total > 0 else 0.0
    
    # 输出结果
    print(f"\n{'='*50}")
    print(f"评测结果:")
    print(f"任务: {task_name}/{task_subdir}")
    print(f"模型: {model_name}")
    print(f"总样本数: {total}")
    print(f"正确: {correct}")
    print(f"错误: {total - correct - invalid}")
    print(f"无效输出: {invalid}")
    print(f"准确率: {acc0:.4f}")
    print(f"{'='*50}")
    
    # 保存结果（参考 test_gsm8k.py）
    TIME = datetime.now().strftime("%m-%d-%H-%M")
    model_name_clean = model_name.replace('/', '').replace('\\', '')
    output_file = f"{model_name_clean}-{task_name}-{task_subdir}-{TIME}.json"
    
    results = {
        "model_name": model_name,
        "task_name": task_name,
        "task_subdir": task_subdir,
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct - invalid,
        "invalid": invalid,
        "accuracy": acc0,
        "answers": answers,
        "acc_list": accs
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存到: {output_file}")
