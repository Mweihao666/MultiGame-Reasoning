from dataclasses import dataclass
from typing import Dict, List

@dataclass
class NashEnvConfig:
    lo_action_name: str = "A"
    hi_action_name: str = "B"

    action_space_start: int = 1

    payoff_matrix_p1: List[List[float]] = None
    payoff_matrix_p2: List[List[float]] = None

    render_mode: str = "text"

    action_lookup: Dict[int, str] = None
