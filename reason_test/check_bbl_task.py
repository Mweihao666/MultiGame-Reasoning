#!/usr/bin/env python3
"""
BIG-bench Lite 任务数据格式检查脚本
用于验证任务数据是否正确加载，并显示任务的基本信息
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def find_bigbench_task(task_name: str, task_dir: Optional[str] = None, 
                      bigbench_root: Optional[str] = None) -> Path:
    """查找 BIG-bench 任务文件（与 bbl_dl.py 中的函数相同）"""
    import os
    
    if task_dir:
        task_json = Path(task_dir) / "task.json"
        if task_json.exists():
            return task_json
        raise FileNotFoundError(f"任务文件未找到: {task_json}")
    
    possible_roots = []
    if bigbench_root:
        possible_roots.append(Path(bigbench_root))
    
    if "BIGBENCH_ROOT" in os.environ:
        possible_roots.append(Path(os.environ["BIGBENCH_ROOT"]))
    
    possible_roots.extend([
        Path("/root/autodl-tmp/bigbench"),
        Path("/root/autodl-tmp/BIG-bench"),
        Path("./bigbench"),
        Path("../bigbench"),
    ])
    
    for root in possible_roots:
        task_json = root / "benchmark_tasks" / task_name / "task.json"
        if task_json.exists():
            return task_json
    
    raise FileNotFoundError(
        f"未找到任务 {task_name} 的 task.json 文件。\n"
        f"请确保 BIG-bench 仓库已克隆，或使用 --task_dir 指定任务目录。"
    )


def check_task_format(task_data: Dict[str, Any]) -> None:
    """检查任务数据格式并显示信息"""
    print("=" * 60)
    print("任务基本信息")
    print("=" * 60)
    print(f"任务名称: {task_data.get('name', 'N/A')}")
    print(f"任务描述: {task_data.get('description', 'N/A')[:100]}...")
    print(f"关键词: {task_data.get('keywords', [])}")
    print()
    
    # 检查示例数据
    examples = task_data.get('examples', [])
    print(f"示例数量: {len(examples)}")
    
    if not examples:
        print("⚠️  警告: 没有找到示例数据！")
        return
    
    # 显示第一个示例
    print("\n" + "=" * 60)
    print("第一个示例:")
    print("=" * 60)
    first_example = examples[0]
    
    print(f"Input: {first_example.get('input', 'N/A')[:200]}...")
    
    # 判断任务类型
    if 'target_scores' in first_example:
        print("\n任务类型: 多项选择 (Multiple Choice)")
        print(f"选项: {list(first_example['target_scores'].keys())}")
        print(f"正确答案: {[k for k, v in first_example['target_scores'].items() if v == 1]}")
    elif 'target' in first_example:
        print("\n任务类型: 生成式 (Generative)")
        print(f"目标答案: {first_example.get('target', 'N/A')[:200]}...")
    else:
        print("\n⚠️  警告: 无法确定任务类型（既没有 target_scores 也没有 target）")
    
    # 检查任务前缀
    task_prefix = task_data.get('task_prefix', '')
    if task_prefix:
        print(f"\n任务前缀: {task_prefix[:200]}...")
    else:
        print("\n任务前缀: 无")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("数据统计:")
    print("=" * 60)
    
    if 'target_scores' in first_example:
        # 多项选择任务统计
        all_choice_counts = {}
        correct_counts = {}
        for ex in examples:
            if 'target_scores' in ex:
                for choice, score in ex['target_scores'].items():
                    all_choice_counts[choice] = all_choice_counts.get(choice, 0) + 1
                    if score == 1:
                        correct_counts[choice] = correct_counts.get(choice, 0) + 1
        
        print(f"选项分布:")
        for choice, count in sorted(all_choice_counts.items()):
            correct = correct_counts.get(choice, 0)
            print(f"  {choice}: {count} 次 (正确答案 {correct} 次)")
    else:
        # 生成式任务统计
        target_lengths = []
        for ex in examples:
            if 'target' in ex:
                target_lengths.append(len(str(ex['target'])))
        
        if target_lengths:
            avg_length = sum(target_lengths) / len(target_lengths)
            print(f"平均答案长度: {avg_length:.1f} 字符")
            print(f"最短答案: {min(target_lengths)} 字符")
            print(f"最长答案: {max(target_lengths)} 字符")
    
    print("\n✅ 任务数据格式检查完成！")


def main():
    parser = argparse.ArgumentParser(description='检查 BIG-bench Lite 任务数据格式')
    parser.add_argument("--task_name", type=str, required=True, help="任务名称")
    parser.add_argument("--task_dir", type=str, default=None, help="任务数据目录")
    parser.add_argument("--bigbench_root", type=str, default=None, help="BIG-bench 根目录")
    
    args = parser.parse_args()
    
    try:
        task_json_path = find_bigbench_task(args.task_name, args.task_dir, args.bigbench_root)
        print(f"✅ 找到任务文件: {task_json_path}\n")
        
        with open(task_json_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
        
        check_task_format(task_data)
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("\n请确保:")
        print("1. BIG-bench 仓库已克隆")
        print("2. 任务名称正确")
        print("3. 使用 --task_dir 或 --bigbench_root 指定路径")
        exit(1)
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()







