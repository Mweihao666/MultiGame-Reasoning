# BIG-bench Lite (BBL) 评测指南

## 概述

本指南说明如何使用 `bbl_dl.py` 脚本评测 BIG-bench Lite 任务。

## 前置准备

### 1. 获取 BIG-bench 任务数据

BIG-bench Lite 任务数据需要从官方仓库获取：

```bash
# 克隆 BIG-bench 仓库
git clone https://github.com/google/BIG-bench.git
cd BIG-bench
```

任务数据位于：`bigbench/benchmark_tasks/{task_name}/task.json`

### 2. 推荐任务选择

根据需求，推荐以下非数学、非纯知识类推理任务：

- **logical_deduction**: 逻辑推理任务（生成式）
- **logic_grid_puzzle**: 逻辑网格谜题（生成式）
- **strategyqa**: 策略性问答（多项选择）

## 使用方法

### 基本用法

```bash
python reason_test/bbl_dl.py \
    --task_name logical_deduction \
    --model_path nash-new \
    --model_name nash50
```

### 参数说明

- `--task_name` (必需): BBL 任务名称，如 `logical_deduction`, `logic_grid_puzzle`, `strategyqa`
- `--model_path`: 模型路径（默认: `nash-new`）
- `--model_name`: 模型名称（默认: `nash50`）
- `--task_dir`: 直接指定任务数据目录路径（可选）
- `--bigbench_root`: 指定 BIG-bench 仓库根目录（可选）

### 指定任务数据路径

如果任务数据不在默认位置，可以使用以下方式指定：

**方式1：指定任务目录**
```bash
python reason_test/bbl_dl.py \
    --task_name logical_deduction \
    --task_dir /path/to/bigbench/benchmark_tasks/logical_deduction \
    --model_path nash-new \
    --model_name nash50
```

**方式2：指定 BIG-bench 根目录**
```bash
python reason_test/bbl_dl.py \
    --task_name logical_deduction \
    --bigbench_root /path/to/BIG-bench \
    --model_path nash-new \
    --model_name nash50
```

**方式3：使用环境变量**
```bash
export BIGBENCH_ROOT=/path/to/BIG-bench
python reason_test/bbl_dl.py \
    --task_name logical_deduction \
    --model_path nash-new \
    --model_name nash50
```

## 输出结果

评测完成后，脚本会：

1. **保存详细结果** 到 `reason_test/results/{model_name}-{task_name}-{timestamp}.json`
   - 包含所有样本的答案和准确率
   
2. **保存日志** 到 `reason_test/{task_name}-log.txt`
   - 记录模型名称、准确率等统计信息

3. **控制台输出** 评测摘要
   - 总样本数、正确数、错误数、无效输出数、准确率

## 任务类型支持

脚本自动识别任务类型：

- **生成式任务 (generative)**: 模型需要生成文本答案
  - 答案提取：从 `<answer>` 标签中提取
  - 评估方式：与正确答案进行字符串匹配
  
- **多项选择任务 (multiple_choice)**: 模型需要从选项中选择
  - 答案提取：提取选项字母（A, B, C, D 等）
  - 评估方式：与正确答案选项比较

## 示例

### 评测 logical_deduction 任务

```bash
python reason_test/bbl_dl.py \
    --task_name logical_deduction \
    --model_path tictactoe \
    --model_name game100
```

### 评测 strategyqa 任务

```bash
python reason_test/bbl_dl.py \
    --task_name strategyqa \
    --model_path nash-new \
    --model_name nash50
```

## 注意事项

1. **数据准备**: 确保 BIG-bench 任务数据已下载并放置在正确位置
2. **模型路径**: 确保模型路径和名称正确，模型需要支持 vLLM 加载
3. **GPU 资源**: 脚本默认使用 GPU 0，如需修改请编辑 `load_llm()` 函数中的 `CUDA_VISIBLE_DEVICES`
4. **批处理大小**: 默认 `batch_size=16`，可根据 GPU 内存调整

## 故障排除

### 找不到任务文件

如果遇到 `FileNotFoundError`，请检查：

1. BIG-bench 仓库是否已克隆
2. 任务名称是否正确（区分大小写）
3. 使用 `--task_dir` 或 `--bigbench_root` 指定正确路径

### 模型加载失败

如果模型加载失败，请检查：

1. 模型路径是否正确
2. vLLM 是否已正确安装
3. GPU 内存是否充足

## 与现有评测流程的集成

`bbl_dl.py` 遵循了 `reason_test/` 文件夹中的代码结构：

- 使用 vLLM 加载模型（参考 `math_dl.py`, `mmlu_dl.py`）
- 批量生成答案
- 提取答案并计算准确率
- 保存结果到 JSON 和日志文件

可以轻松集成到现有的评测流程中。







