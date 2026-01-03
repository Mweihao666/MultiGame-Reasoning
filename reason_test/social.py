import json
from collections import Counter
import argparse
import tqdm
import random
import re
import os
import logging
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragen.env.base import EnvPlayer

root_path = '/root/autodl-tmp'
batch_size = 16

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="google/gemini-2.5-flash-nothinking", 
                    help="模型名称，支持 gemini (google/gemini-2.5-flash-nothinking) 或 gpt-4o")
parser.add_argument("--max_samples", type=int, default=None, help="最大测试样本数（None表示使用全部样本）")
parser.add_argument("--max_tokens", type=int, default=600, help="最大生成 token 数")
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
                print(f"  API 返回空结果，{wait_time} 秒后重试 (第 {retries} 次)...")
                time.sleep(wait_time)
        except Exception as e:
            retries += 1
            wait_time = min(2 ** retries, 30)  # 指数退避，最多等待30秒
            print(f"  API 调用失败: {e}，{wait_time} 秒后重试 (第 {retries} 次)...")
            time.sleep(wait_time)

# jsonl不能直接读取，每一行是一个单独的json对象
# test_data = json.load(open(f"{root_path}/reasoning/socialiqa-train-dev/dev.jsonl"))
social = []
with open(f"{root_path}/reasoning/socialiqa-train-dev/dev.jsonl", "r") as f:
    for line in f:
        social.append(json.loads(line))
with open(f"{root_path}/reasoning/socialiqa-train-dev/dev-labels.lst", "r") as f:
    label_all = f.read().splitlines()
label_all = [int(l) - 1 for l in label_all]

def load_llm():
    """兼容性函数，返回 env_player 和 None"""
    return env_player, None


def reformat_prompt(prompt0, choice):
    # 将prompt句子末尾的 Let\'s think step by step and output the final answer after "####".
    # 替换为Let\'s think step by step and output your think and final answer in this format: 
    # <think> [your thought] </think> <answer> [your answer] </answer>
    formatted_question = (
        prompt0.strip() + "\n"
        + f"A. {choice[0]}\n"
        + f"B. {choice[1]}\n"
        + f"C. {choice[2]}\n"
    )
    prompt = formatted_question + "Let\'s think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens)."
    # 注意：chat template 现在由 model_adapter 处理
    return prompt


def extract_solution(solution_str):
    pattern = r"<answer>(.*)</answer>"
    # 使用 re.search() 提取最后一个数字
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        last_number = match.group(1)  # 提取匹配到的最后一个数字
        solu_strict = last_number
    else:
        solu_strict = None 
    return solu_strict


def extract_choice(text: str):
    """
    从文本中提取第一个A-D字母（句首答案），允许格式：
    - A
    - A.
    - A.hello
    - A something
    不匹配：
    - ADHD, BADGE 等单词中嵌入的情况
    """
    text = text.strip()
    match = re.search(r'(?<![A-Za-z])([A-D])(\.|(?:\s|$))', text)
    if match:
        return match.group(1).upper()
    return None


def test_social(llm, sampling_params, social, ground_truth, max_samples=None):
    answers = []
    acc_list = []
    total_samples = min(len(social), max_samples) if max_samples else len(social)
    for i in tqdm.trange(0, total_samples, batch_size):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        data = social[i: i + batch_size]
        labels = ground_truth[i: i + batch_size]
        '''
        doc_to_text: "{{context}}\nQuestion: {{question}}\nAnswer:"  
        doc_to_target: "{{answer}}"  # 正确答案的索引  
        doc_to_choice: "{{[answerA, answerB, answerC]}}"  # 三个选项
        '''
        prompts = [reformat_prompt(data[j]['context'] + '\nQuestion: ' + 
                                  data[j]['question'], 
                                 [data[j]['answerA'], data[j]['answerB'], data[j]['answerC']]) for j in range(len(data))]
        
        # 使用 EnvPlayer 逐个生成（API 调用不支持批量，llm_output 内部已处理重试）
        outputs = []
        for prompt in prompts:
            output_text = llm_output(prompt)
            # 包装成类似 vLLM 输出的格式，保持兼容性
            class FakeOutput:
                def __init__(self, text):
                    self.text = text
            class FakeRequestOutput:
                def __init__(self, text):
                    self.outputs = [FakeOutput(text)]
            outputs.append(FakeRequestOutput(output_text))
        
        for j, out in enumerate(outputs):
            # answer, choices, subject
            solution = extract_solution(out.outputs[0].text)
            answers.append(out.outputs[0].text)
            if solution is None:
                choice = None
            else:
                choice = extract_choice(solution)
                answer = ['A', 'B', 'C'][labels[j]]
                choice = (choice == answer)
            acc_list.append(choice)
    return answers, acc_list


def save_sample_results(model_name, acc_list, answers, social, ground_truth, 
                       num_samples=5, output_dir="reason_test/results"):
    """
    保存部分正确和错误的测试结果，便于分析
    
    Args:
        model_name: 模型名称
        acc_list: 准确率列表、不区分类别
        answers: 模型生成的答案列表
        social: 测试数据集
        ground_truth: 正确答案列表
        num_samples: 每种类型保存的样本数量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    output_file = f"{output_dir}/{model_name}-socialIQA-{time_str}.json"
    
    # 收集正确和错误的样本索引
    correct_indices = []
    incorrect_indices = []
    invalid_indices = []
    
    # 遍历所有类别和样本
    idx = 0
    for result in acc_list:
        if result is None:
            invalid_indices.append(idx)
        elif result == 1:
            correct_indices.append(idx)
        else:
            incorrect_indices.append(idx)
        idx += 1
    
    # 限制样本数量
    if len(correct_indices) > num_samples:
        correct_indices = correct_indices[:num_samples]
    if len(incorrect_indices) > num_samples:
        incorrect_indices = incorrect_indices[:num_samples]
    if len(invalid_indices) > num_samples:
        invalid_indices = invalid_indices[:num_samples]
    
    # 准备保存的数据
    samples = {
        "model_name": model_name,
        "correct_samples": [],
        "incorrect_samples": [],
        "invalid_samples": []
    }
    
    # 添加正确样本
    for idx in correct_indices:
        # 获取对应的题目信息
        data_idx = idx
        samples["correct_samples"].append({
            "question": social[data_idx]['context'] + '\nQuestion: ' + social[data_idx]['question'],
            "choices": [social[data_idx]['answerA'], social[data_idx]['answerB'], social[data_idx]['answerC']],
            "model_answer": answers[idx],
            "ground_truth": ['A', 'B', 'C'][ground_truth[data_idx]],
        })
    
    # 添加错误样本
    for idx in incorrect_indices:
        # 获取对应的题目信息
        data_idx = idx
        samples["incorrect_samples"].append({
            "question": social[data_idx]['context'] + '\nQuestion: ' + social[data_idx]['question'],
            "choices": [social[data_idx]['answerA'], social[data_idx]['answerB'], social[data_idx]['answerC']],
            "model_answer": answers[idx],
            "ground_truth": ['A', 'B', 'C'][ground_truth[data_idx]],
        })
    
    # 添加无效样本
    for idx in invalid_indices:
        # 获取对应的题目信息
        data_idx = idx
        samples["invalid_samples"].append({
            "question": social[data_idx]['context'] + '\nQuestion: ' + social[data_idx]['question'],
            "choices": [social[data_idx]['answerA'], social[data_idx]['answerB'], social[data_idx]['answerC']],
            "model_answer": answers[idx],
            "ground_truth": ['A', 'B', 'C'][ground_truth[data_idx]],
        })
    
    # 保存到文件
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=4)
    
    print(f" Sample results saved to {output_file}")
    print(f"Saved {len(correct_indices)} correct, {len(incorrect_indices)} incorrect and {len(invalid_indices)} invalid samples")


def save_log(model_name, accs_list, output_file="reason_test/socialIQA-log.txt"):  
    acc0_strict = accs_list.count(1) / len(accs_list)
    invalid_strict = accs_list.count(None)

    # 写入日志文件
    with open(output_file, "a") as f:
        f.write(f"\n=== Model: {model_name} ===\n")
        f.write("socialiqa dev test set\n")
        f.write("strict mode\n")
        f.write(f"total acc: {acc0_strict:.4f}\n")
        f.write(f"invalid output: {invalid_strict}\n")

    print(f" Log saved to {output_file}")


if __name__ == '__main__':
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    logging.getLogger("vllm").setLevel(logging.ERROR)
    path0 = f'/root/autodl-tmp/reasoning'
    llm, sampling_params = load_llm()
    answers, acc_list = test_social(llm, sampling_params, social, label_all, max_samples=args.max_samples)
    # 保存测试结果到日志文件
    save_log(model_name, acc_list)
    # 保存部分测试结果，便于分析
    save_sample_results(model_name, acc_list, answers, social, label_all, num_samples=20)
