# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import time
import datasets
import tqdm
from math_verify import parse, verify
import os, logging
import argparse
import random
from model_adapter import create_model_adapter

root_path = '/root/autodl-tmp'  # '/data1/lvnuoyan' 
batch_size = 16
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default='vllm', 
                    choices=['deepseek', 'gemini', 'bbl-lite', 'vllm'],
                    help="模型类型：deepseek, gemini, bbl-lite, vllm")
parser.add_argument("--model_path", type=str, default="tictactoe")
parser.add_argument("--model_name", type=str, default="game50")
parser.add_argument("--port", type=str, default="2100", help="vLLM 服务端口（仅用于 vllm 和 bbl-lite 类型）")
parser.add_argument("--max_samples", type=int, default=None, help="最大测试样本数（None表示使用全部样本）")
args = parser.parse_args()
model_type = args.model_type
model_path = args.model_path
model_name = args.model_name
port = args.port
time_str = time.strftime("%m-%d-%H-%M", time.localtime())

# 创建模型适配器
model_adapter = create_model_adapter(
    model_type=model_type,
    model_name=model_name,
    model_path=model_path,
    port=port
)

def load_llm():
    """兼容性函数，返回 model_adapter 和 None（sampling_params 不再需要）"""
    return model_adapter, None


def reformat_prompt(prompt0):

    prompt = prompt0.replace("Let\'s think step by step and output the final answer after \"####\".", 
                             "Let\'s think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens).")
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


def test_math(llm, sampling_params, math, max_samples=None):
    """测试数学任务，使用 model_adapter 进行 API 调用"""
    accs_strict = []
    accs_flex = []
    answers = []
    total_samples = min(len(math['test']), max_samples) if max_samples else len(math['test'])
    for i in tqdm.trange(0, total_samples, batch_size):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        data = math['test'][i: i + batch_size]
        prompts = [reformat_prompt(data['prompt'][j][0]['content']) for j in range(len(data['prompt']))]
        
        # 使用 model_adapter 逐个生成（API 调用不支持批量）
        outputs = []
        for prompt in prompts:
            try:
                output_text = llm.generate(
                    prompt=prompt,
                    max_tokens=600,
                    temperature=0.5,
                    use_chat_template=True
                )
                # 包装成类似 vLLM 输出的格式，保持兼容性
                class FakeOutput:
                    def __init__(self, text):
                        self.text = text
                class FakeRequestOutput:
                    def __init__(self, text):
                        self.outputs = [FakeOutput(text)]
                outputs.append(FakeRequestOutput(output_text))
            except Exception as e:
                print(f" 生成失败: {e}")
                outputs.append(FakeRequestOutput(""))
        
        for j, out in enumerate(outputs):
            solu_strict, solu_flex = extract_solution(out.outputs[0].text)
            answers.append(out.outputs[0].text)
            # print(outputs[0].outputs[0].text)
            ground_truth = parse(data['reward_model'][j]['ground_truth'])
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


def save_log(model_name, accs_strict, accs_flex, output_file="reason_test/gsm8k-log.txt"):
    acc0_strict = accs_strict.count(1) / len(accs_strict)
    invalid_strict = accs_strict.count(None)
    acc0_flex = accs_flex.count(1) / len(accs_flex)

    # 写入日志文件
    with open(output_file, "a") as f:
        f.write(f"\n=== Model: {model_name} ===\n")
        f.write("gsm8k test set\n")
        f.write("strict mode\n")
        f.write(f"total acc: {acc0_strict:.4f}\n")
        f.write(f"invalid output: {invalid_strict}\n")
        f.write("flexible mode\n")
        f.write(f"total acc: {acc0_flex:.4f}\n")

    print(f" Log saved to {output_file}")


def save_sample_results(model_name, accs_strict, accs_flex, answers, math_data, 
                       num_samples=5, output_dir="reason_test/results"):
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
    output_file = f"{output_dir}/{model_name}-gsm8k-{time_str}.json"
    
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
    # invalid担心数量不够——先判断一下
    if len(invalid_indices) > num_samples:
        invalid_indices = invalid_indices[:num_samples]
    
    # 准备保存的数据
    samples = {
        "model_name": model_name,
        "correct_samples": [],
        "incorrect_samples": [],
        "invalid_samples": [],
    }
    
    # 添加正确样本
    for idx in correct_indices:
        samples["correct_samples"].append({
            "question": math_data['test']['prompt'][idx][0]['content'],
            "model_answer": answers[idx],
            "ground_truth": math_data['test']['reward_model'][idx]['ground_truth'],
        })
    
    # 添加错误样本
    for idx in incorrect_indices:
        samples["incorrect_samples"].append({
            "question": math_data['test']['prompt'][idx][0]['content'],
            "model_answer": answers[idx],
            "ground_truth": math_data['test']['reward_model'][idx]['ground_truth'],
        })
    
    # 添加invalid样本
    for idx in invalid_indices:
        samples["invalid_samples"].append({
            "question": math_data['test']['prompt'][idx][0]['content'],
            "model_answer": answers[idx],
            "ground_truth": math_data['test']['reward_model'][idx]['ground_truth'],
        })
    
    
    # 保存到文件
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=4)
    
    print(f"✅ Sample results saved to {output_file}")
    print(f"Saved {len(correct_indices)} correct, {len(incorrect_indices)} incorrect samples and {len(invalid_indices)} invalid samples.")


if __name__ == '__main__':
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    logging.getLogger("vllm").setLevel(logging.ERROR)
    path0 = f'/root/autodl-tmp/reasoning'
    math = datasets.load_dataset("parquet", 
                  data_files={'train': path0 + '/gsm8k/train.parquet', 'test': path0 + '/gsm8k/test.parquet'})
    # print(math['test']['prompt'][0])
    llm, sampling_params = load_llm()
    # exit(0)
    accs_strict, accs_flex, answers = test_math(llm, sampling_params, math, max_samples=args.max_samples)
    acc0 = accs_strict.count(1) / len(accs_strict)
    print('model:', model_name)
    print('gsm8k test set')
    print('-----strict mode-----')
    print('total acc:', format(acc0, '.4f'))
    print('invalid output:', accs_strict.count(None))
    print('----flexible mode----')
    acc0 = accs_flex.count(1) / len(accs_flex)
    print('total acc:', format(acc0, '.4f'))
    # 写入日志文件
    save_log(model_name, accs_strict, accs_flex)
    # 保存部分测试结果，便于分析
    save_sample_results(model_name, accs_strict, accs_flex, answers, math, num_samples=20)
