from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

@dataclass
class ComposeNewConfig:
    base_env_list: List[str] = field(
        default_factory=lambda: ["math_lv3to5", "nash_new", "tictactoe", "undercover"]
    )
    player_nums: List[int] = field(
        default_factory=lambda: [0, 0, 2, 4]
    )
    player_infos: List[List[Dict[str, Any]]] = field(
        # default_factory=lambda: [{'model_name': 'deepseek'}]
        # default_factory=lambda: [{'model_name': 'tictactoe/grpo/game_40', 'port': '4040'}]
        default_factory=lambda: 
            # [{'model_name': 'google/gemini-2.5-flash'}, # {'model_name': 'google/gemini-2.5-flash'},
            #  {'model_name': 'google/gemini-2.5-flash'},{'model_name': 'google/gemini-2.5-flash'},]
            # [{"model_name": "x-ai/grok-4-fast"}, {"model_name": "x-ai/grok-4-fast"},
            #  {"model_name": "x-ai/grok-4-fast"}, {"model_name": "x-ai/grok-4-fast"},]
            [[{'': ''}],[{'': ''}],
             [{'model_name': 'Qwen3-14B', 'port': '1414'}],
             [{'model_name': 'Qwen3-14B', 'port': '1414'},
              {'model_name': 'Qwen3-14B', 'port': '1414'},
              {'model_name': 'Qwen3-14B', 'port': '1414'}]]
            # [{'model_name': "gemini-2.5-flash-nothinking"}]]
            # [{'model_name': 'Qwen3-14B', 'port': '1414'},
            #  {'model_name': 'Qwen3-14B', 'port': '1414'},
            #  {'model_name': 'Qwen3-14B', 'port': '1414'}]]
    )
    reward_improve: bool = False
    k: int = 4 # 计算成功率使用的最近k个epoch
    max_num: int = 500 # 计算成功率的时候使用的最近多少个success数据
    mode: str = 'test'  # train or test for mix env.
    seed: int = 123
    render_mode: str = 'text'