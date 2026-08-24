"""The ECS core: entity ids, versioned component registry, systems stepped with actions.

Pinned agency-layer constraints (the environment reuses this core):
  - `World.step(actions) -> events`: dynamics run without any rendering.
  - Actions come in from outside; the data generator scripts them, an agent may later.
  - `World.serialize()` is a stable, sorted byte encoding of the full state.
  - Determinism: everything random flows through `World.rng` seeded at construction.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

SCHEMA_VERSION = 1


@dataclass
class Event:
    """One thing that happened: a verb with participants, for renderers and logs."""
    t: int
    kind: str
    data: Dict[str, Any]


class World:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.seed = seed
        self.t = 0
        self._next_eid = 0
        # component store: name -> {eid -> dataclass instance}
        self.components: Dict[str, Dict[int, Any]] = {}
        self.systems: List[Callable[["World", List[Dict[str, Any]]], List[Event]]] = []
        self.names: Dict[int, str] = {}  # display name per entity (locale-independent key)

    # --- entities/components ------------------------------------------------------
    def spawn(self, name: str) -> int:
        eid = self._next_eid
        self._next_eid += 1
        self.names[eid] = name
        return eid

    def add(self, eid: int, comp: Any) -> None:
        self.components.setdefault(type(comp).__name__, {})[eid] = comp

    def get(self, eid: int, comp_type: type) -> Optional[Any]:
        return self.components.get(comp_type.__name__, {}).get(eid)

    def query(self, *comp_types: type):
        """Iterate (eid, comp, ...) over entities holding every listed component."""
        if not comp_types:
            return
        stores = [self.components.get(c.__name__, {}) for c in comp_types]
        for eid in sorted(stores[0]):
            if all(eid in s for s in stores[1:]):
                yield (eid, *(s[eid] for s in stores))

    # --- dynamics -----------------------------------------------------------------
    def step(self, actions: List[Dict[str, Any]]) -> List[Event]:
        """Advance one tick: every system sees the same action list, events accumulate."""
        events: List[Event] = []
        for system in self.systems:
            events.extend(system(self, actions))
        self.t += 1
        return events

    # --- state --------------------------------------------------------------------
    def serialize(self) -> bytes:
        """Stable byte encoding of the full state (sorted keys throughout)."""
        state = {
            "schema": SCHEMA_VERSION,
            "t": self.t,
            "names": {str(k): v for k, v in sorted(self.names.items())},
            "components": {
                cname: {str(eid): asdict(comp) for eid, comp in sorted(store.items())}
                for cname, store in sorted(self.components.items())
            },
        }
        return json.dumps(state, sort_keys=True, ensure_ascii=False).encode("utf-8")
