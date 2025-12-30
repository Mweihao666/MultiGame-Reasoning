#!/bin/bash

# 快速测试脚本 - 每个benchmark只测试部分样本
# 使用方法: bash reason_test/run_quick_test.sh gemini gemini-2.5-flash-nothinking 50 10

MODEL_TYPE=${1:-gemini}
MODEL_NAME=${2:-gemini-2.5-flash-nothinking}
MAX_SAMPLES=${3:-}  # 最大测试样本数（空值表示不限制，使用全部样本）
MAX_ROUNDS=${4:-}   # 最大测试轮数（空值表示不限制，使用默认轮数）

echo "===== 快速测试开始 ====="
echo "模型类型: $MODEL_TYPE"
echo "模型名称: $MODEL_NAME"
echo "最大样本数: $MAX_SAMPLES"
echo "最大轮数: $MAX_ROUNDS"
echo ""

# 游戏测试
echo "--- 游戏测试 ---"
if [ -n "$MAX_ROUNDS" ]; then
    python reason_test/nash-new.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --max_rounds $MAX_ROUNDS
    python reason_test/tictactoe.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --max_rounds $MAX_ROUNDS
    python reason_test/undercover.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --max_rounds $MAX_ROUNDS
else
    python reason_test/nash-new.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
    python reason_test/tictactoe.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
    python reason_test/undercover.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
fi

# MMLU测试
echo "--- MMLU测试 ---"
if [ -n "$MAX_SAMPLES" ]; then
    python reason_test/mmlu_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --max_samples $MAX_SAMPLES
    python reason_test/mmlu_pro_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --max_samples $MAX_SAMPLES
else
    python reason_test/mmlu_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
    python reason_test/mmlu_pro_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
fi

# Math测试
echo "--- Math测试 ---"
if [ -n "$MAX_SAMPLES" ]; then
    python reason_test/math_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --max_samples $MAX_SAMPLES
    python reason_test/lv3to5_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --max_samples $MAX_SAMPLES
else
    python reason_test/math_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
    python reason_test/lv3to5_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
fi

# Social测试
echo "--- Social测试 ---"
if [ -n "$MAX_SAMPLES" ]; then
    python reason_test/social.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --max_samples $MAX_SAMPLES
else
    python reason_test/social.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
fi

# Common测试
echo "--- Common测试 ---"
if [ -n "$MAX_SAMPLES" ]; then
    python reason_test/common.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --max_samples $MAX_SAMPLES
else
    python reason_test/common.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
fi

# BBL测试
echo "--- BBL测试 ---"
if [ -n "$MAX_SAMPLES" ]; then
    python reason_test/bbl_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --task_name logical_deduction --task_subdir five_objects --max_samples $MAX_SAMPLES
else
    python reason_test/bbl_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --task_name logical_deduction --task_subdir five_objects
fi

# GSM8K测试
echo "--- GSM8K测试 ---"
if [ -n "$MAX_SAMPLES" ]; then
    python reason_test/test_gsm8k.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --max_samples $MAX_SAMPLES
else
    python reason_test/test_gsm8k.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
fi

echo ""
echo "✅ 快速测试完成"
