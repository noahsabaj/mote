"""Walking nested tensor state.

`InferenceState`, a Mamba-3 cache tuple, a feedback pass's front half — every one is tensors nested
in lists, tuples, NamedTuples and dataclasses, and three places used to walk them with three private
recursions (`hnet._map_state`, `feedback.detach_tree`, `prefix_cache.state_nbytes`). One walk.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Callable, Iterator, Tuple, Type

import torch


def map_tree(o: Any, fn: Callable[[Any], Any], atoms: Tuple[Type, ...] = ()) -> Any:
    """`o` rebuilt with `fn` applied to every tensor. `atoms` are extra types handed to `fn` whole
    instead of walked (a shared arena reference, say). Scalars, strings and None pass through;
    anything else is deep-copied."""
    if isinstance(o, torch.Tensor) or (atoms and isinstance(o, atoms)):
        return fn(o)
    if isinstance(o, list):
        return [map_tree(x, fn, atoms) for x in o]
    if isinstance(o, tuple):
        parts = [map_tree(x, fn, atoms) for x in o]
        return type(o)(*parts) if hasattr(o, "_fields") else tuple(parts)
    if o is None or isinstance(o, (int, float, bool, str)):
        return o
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return type(o)(**{f.name: map_tree(getattr(o, f.name), fn, atoms) for f in dataclasses.fields(o)})
    return copy.deepcopy(o)


def iter_tree(o: Any) -> Iterator[torch.Tensor]:
    """Every tensor in `o`, depth first."""
    if isinstance(o, torch.Tensor):
        yield o
    elif isinstance(o, (list, tuple)):
        for x in o:
            yield from iter_tree(x)
    elif dataclasses.is_dataclass(o) and not isinstance(o, type):
        for f in dataclasses.fields(o):
            yield from iter_tree(getattr(o, f.name))


def tree_nbytes(o: Any) -> int:
    return sum(t.numel() * t.element_size() for t in iter_tree(o))


def detach_tree(o: Any) -> Any:
    return map_tree(o, lambda t: t.detach())
