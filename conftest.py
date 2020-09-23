import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
