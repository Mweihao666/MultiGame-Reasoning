#!/bin/bash

# 完整测试脚本
# 使用方法: bash reason_test/all_test.sh gemini gemini-2.5-flash-nothinking

MODEL_TYPE=${1:-gemini}
MODEL_NAME=${2:-gemini-2.5-flash-nothinking}

echo "===== 完整测试开始 ====="
echo "模型类型: $MODEL_TYPE"
echo "模型名称: $MODEL_NAME"
echo ""

# 游戏测试
echo "--- 游戏测试 ---"
python reason_test/nash-new.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
python reason_test/tictactoe.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
python reason_test/undercover.py --model_type $MODEL_TYPE --model_name $MODEL_NAME

# MMLU测试
echo "--- MMLU测试 ---"
python reason_test/mmlu_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
python reason_test/mmlu_pro_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME

# Math测试
echo "--- Math测试 ---"
python reason_test/math_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME
python reason_test/lv3to5_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME

# Social测试
echo "--- Social测试 ---"
python reason_test/social.py --model_type $MODEL_TYPE --model_name $MODEL_NAME

# Common测试
echo "--- Common测试 ---"
python reason_test/common.py --model_type $MODEL_TYPE --model_name $MODEL_NAME

# BBL测试
echo "--- BBL测试 ---"
python reason_test/bbl_dl.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --task_name logical_deduction --task_subdir five_objects

# GSM8K测试
echo "--- GSM8K测试 ---"
python reason_test/test_gsm8k.py --model_type $MODEL_TYPE --model_name $MODEL_NAME

echo ""
echo "✅ 完整测试完成"
