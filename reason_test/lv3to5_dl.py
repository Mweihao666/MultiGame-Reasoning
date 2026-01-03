import json
import re
import time
import datasets
import tqdm
from math_verify import parse, verify
import os, logging
import argparse
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragen.env.base import EnvPlayer

root_path = '/root/autodl-tmp'  # '/data1/lvnuoyan' 
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
time_str = time.strftime("%m-%d-%H-%M", time.localtime())

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
                print(f"⚠️  API 返回空结果，{wait_time} 秒后重试 (第 {retries} 次)...")
                time.sleep(wait_time)
        except Exception as e:
            retries += 1
            wait_time = min(2 ** retries, 30)  # 指数退避，最多等待30秒
            print(f"⚠️  API 调用失败: {e}，{wait_time} 秒后重试 (第 {retries} 次)...")
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


def extract_solution(solution_str):
    pattern = r"<answer>(.*)</answer>"
    # 使用 re.search() 提取最后一个数字
    match = re.search(pattern, solution_str, re.DOTALL)
    solu_flex = solution_str
    if match:
        last_number = match.group(1)  # 提取匹配到的最后一个数字
        solu_strict = last_number
        solu_flex = last_number
    else:
        solu_strict = None 
    return solu_strict, solu_flex


def save_log(model_name, accs_strict, accs_flex, output_file="reason_test/math500-log.txt"):
    acc0_strict = accs_strict.count(1) / len(accs_strict)
    invalid_strict = accs_strict.count(None)
    acc0_flex = accs_flex.count(1) / len(accs_flex)

    # 写入日志文件
    with open(output_file, "a") as f:
        f.write(f"\n=== Model: {model_name} ===\n")
        f.write("math 500 test set\n")
        f.write("strict mode\n")
        f.write(f"total acc: {acc0_strict:.4f}\n")
        f.write(f"invalid output: {invalid_strict}\n")
        f.write("flexible mode\n")
        f.write(f"total acc: {acc0_flex:.4f}\n")

    print(f"✅ Log saved to {output_file}")


def test_math(llm, sampling_params, math, max_samples=None):
    accs_strict = []
    accs_flex = []
    answers = []
    total_samples = min(len(math), max_samples) if max_samples else len(math)
    for i in tqdm.trange(0, total_samples, batch_size):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        data = math[i: i + batch_size]
        prompts = [reformat_prompt(data['extra_info'][j]['question']) for j in range(len(data['extra_info']))]
        
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
            solu_strict, solu_flex = extract_solution(out.outputs[0].text)
            answers.append(out.outputs[0].text)
            # print(outputs[0].outputs[0].text)
            ground_truth = parse(data['extra_info'][j]['answer'])
            correct_strict = verify(parse(solu_strict), ground_truth)
            if solu_strict is None:
                accs_strict.append(None)
            elif correct_strict:
                accs_strict.append(1)
            else:
                # print(solution, ground_truth)
                accs_strict.append(0)
            correct_flex = verify(parse(solu_flex), ground_truth)
            if solu_flex is None:
                accs_flex.append(None)
            elif correct_flex:
                accs_flex.append(1)
            else:
                accs_flex.append(0)
    return accs_strict, accs_flex, answers


def save_sample_results(model_name, accs_strict, accs_flex, answers, math_data, 
                       num_samples=20, output_dir="reason_test/results"):
    """
    保存部分正确和错误的测试结果，便于分析
    
    Args:
        model_name: 模型名称
        accs_strict: 严格模式下的准确率列表
        accs_flex: 灵活模式下的准确率列表
        answers: 模型生成的答案列表
        math_data: 测试数据集
        num_samples: 每种类型保存的样本数量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    output_file = f"{output_dir}/{model_name}-math500-{time_str}.json"
    
    # 收集正确和错误的样本索引
    correct_indices = []
    incorrect_indices = []
    invalid_indices = []
    
    for i, (acc_strict, acc_flex) in enumerate(zip(accs_strict, accs_flex)):
        if acc_strict is None:
            invalid_indices.append(i)
        elif acc_strict == 1:
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)
    
    # 限制样本数量
    correct_indices = correct_indices[:num_samples]
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
        samples["correct_samples"].append({
            "question": math_data['extra_info'][idx]['question'],
            "model_answer": answers[idx],
            "ground_truth": math_data['extra_info'][idx]['answer'],
        })
    
    # 添加错误样本
    for idx in incorrect_indices:
        samples["incorrect_samples"].append({
            "question": math_data['extra_info'][idx]['question'],
            "model_answer": answers[idx],
            "ground_truth": math_data['extra_info'][idx]['answer'],
        })

    # 添加invalid样本
    for idx in invalid_indices:
        samples["invalid_samples"].append({
            "question": math_data['extra_info'][idx]['question'],
            "model_answer": answers[idx],
            "ground_truth": math_data['extra_info'][idx]['answer'],
        })
    
    # 保存到文件
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=4)
    
    print(f"✅ Sample results saved to {output_file}")
    print(f"Saved {len(correct_indices)} correct and {len(incorrect_indices)} incorrect samples")


# 复制出问题了，需要改回mathlv3-5的数据集
if __name__ == '__main__':
    path0 = f'/root/autodl-tmp/reasoning'
    # math = datasets.load_dataset("parquet", 
    #               data_files={'train': path0 + '/gsm8k/train.parquet', 'test': path0 + '/gsm8k/test.parquet'})
    # # print(math['test']['prompt'][0])
    math = datasets.load_dataset("parquet", 
                  data_files=path0 + '/SimpleRL-Zoo-Data/simplelr_abel_level3to5/test.parquet')['train']
    llm, sampling_params = load_llm()
    # exit(0)
    accs_strict, accs_flex, answers = test_math(llm, sampling_params, math, max_samples=args.max_samples)
    acc0 = accs_strict.count(1) / len(accs_strict)
    print('model:', model_name)
    print('math lv3-5 test set')
    print('-----strict mode-----')
    print('total acc:', format(acc0, '.4f'))
    print('invalid output:', accs_strict.count(None))
    print('----flexible mode----')
    acc0 = accs_flex.count(1) / len(accs_flex)
    print('total acc:', format(acc0, '.4f'))
    save_log(model_name, accs_strict, accs_flex)
    save_sample_results(model_name, accs_strict, accs_flex, answers, math)
