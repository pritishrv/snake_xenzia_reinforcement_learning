import random

import numpy as np


def setGlobalSeed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
