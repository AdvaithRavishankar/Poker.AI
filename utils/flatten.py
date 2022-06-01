import numpy as np
from gym.spaces import Box, Discrete, Tuple, Dict, MultiBinary, MultiDiscrete


def flatdim(space):
    if isinstance(space, Box):
        return space
    elif isinstance(space, Discrete):
        return space
    elif isinstance(space, Tuple):
        return [flatdim(s) for s in list(space)]
    elif isinstance(space, Dict):
        return [flatdim(s) for s in list(space.values())]
    elif isinstance(space, MultiBinary):
        return [Discrete(1) for _ in range(space.n)]
    elif isinstance(space, MultiDiscrete):
        return space
    else:
        raise NotImplementedError


def flatten_array(container):
    def _flatten_array(container):
        for i in container:
            if isinstance(i, (list, tuple)):
                for j in flatten_array(i):
                    yield j
            else:
                yield i

    return list(_flatten_array(container))


def get_boundary(space, is_low=True):
    def _get_boundary(space, is_low=True):
        for s in space:
            yield s.start if is_low else s.n - 1

    return list(_get_boundary(space, is_low))


def flatten_spaces(space):
    space = flatten_array(flatdim(space))
    low = np.array(get_boundary(space))
    high = np.array(get_boundary(space, is_low=False))
    return Box(low, high, low.shape, dtype=np.int32)
