import gym
import numpy as np
from collections import OrderedDict
from typing import Sequence, Tuple, List

def atariar_info_to_rddlgym(info: dict) -> Sequence[Tuple[str, List[str]]]:
    # FluentParamsList = Sequence[Tuple[str, List[str]]] in pyrddl
    labels = info['labels'] # dict[str, int]
    return OrderedDict([(k, np.array([float(v)])) for k, v in labels.items()])
