# BIG-bench Lite 评测快速开始指南

## 快速开始

### 1. 准备 BIG-bench 数据

```bash
# 克隆 BIG-bench 仓库（如果还没有）
git clone https://github.com/google/BIG-bench.git
cd BIG-bench
```

### 2. 检查任务数据（可选但推荐）

在运行评测前，先检查任务数据是否正确：

```bash
python reason_test/check_bbl_task.py --task_name logical_deduction
```

如果任务数据不在默认位置，可以指定路径：

```bash
python reason_test/check_bbl_task.py \
    --task_name logical_deduction \
    --bigbench_root /path/to/BIG-bench
```

### 3. 运行评测

```bash
python reason_test/bbl_dl.py \
    --task_name logical_deduction \
    --model_path nash-new \
    --model_name nash50
```

## 推荐任务

根据您的需求（非数学、非纯知识类推理任务），推荐以下任务：

1. **logical_deduction** - 逻辑推理（生成式）
2. **logic_grid_puzzle** - 逻辑网格谜题（生成式）
3. **strategyqa** - 策略性问答（多项选择）

## 完整示例

### 示例 1: 评测 logical_deduction

```bash
# 1. 检查数据
python reason_test/check_bbl_task.py --task_name logical_deduction

# 2. 运行评测
python reason_test/bbl_dl.py \
    --task_name logical_deduction \
    --model_path tictactoe \
    --model_name game100 \
    --bigbench_root /root/autodl-tmp/BIG-bench
```

### 示例 2: 评测 strategyqa

```bash
python reason_test/bbl_dl.py \
    --task_name strategyqa \
    --model_path nash-new \
    --model_name nash50
```

## 输出说明

评测完成后，会生成：

1. **详细结果 JSON**: `reason_test/results/{model_name}-{task_name}-{timestamp}.json`
   - 包含所有样本的答案和准确率

2. **日志文件**: `reason_test/{task_name}-log.txt`
   - 记录评测统计信息

3. **控制台输出**: 显示准确率等关键指标

## 常见问题

### Q: 找不到任务文件？

**A:** 确保：
- BIG-bench 仓库已克隆
- 任务名称正确（区分大小写）
- 使用 `--task_dir` 或 `--bigbench_root` 指定正确路径

### Q: 如何指定任务数据路径？

**A:** 三种方式：

1. `--task_dir /path/to/bigbench/benchmark_tasks/logical_deduction`
2. `--bigbench_root /path/to/BIG-bench`
3. 设置环境变量 `export BIGBENCH_ROOT=/path/to/BIG-bench`

### Q: 模型加载失败？

**A:** 检查：
- 模型路径是否正确
- vLLM 是否已安装
- GPU 内存是否充足

## 下一步

评测完成后，可以：

1. 查看详细结果 JSON 文件分析模型表现
2. 对比不同模型的准确率
3. 根据任务特点调整答案提取逻辑（如需要）

## 相关文件

- `bbl_dl.py` - 主评测脚本
- `check_bbl_task.py` - 任务数据检查脚本
- `BBL_EVALUATION_README.md` - 详细文档







