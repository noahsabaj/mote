"""The ECS core on esper 3 (decided 2026-08-24), plus first-class relationships.

Our contract is unchanged and is what the future agency/RL environment consumes:
  - `World.step(actions) -> events`: dynamics run as esper processors, no rendering inside.
  - Actions come in from outside; the generator scripts them, an agent may later.
  - `World.serialize()` is a stable, sorted byte encoding of the full state.
  - Determinism: everything random flows through `World.rng` seeded at construction.

esper keeps global named worlds; each `World` owns one (switched on every call) and deletes it
on `close()`. Relationships are the one idea taken from flecs: `(subject, relation, object)`
triples with reverse and transitive queries, replacing string-valued "pointer" fields.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import esper

SCHEMA_VERSION = 2
_ids = itertools.count()


@dataclass
class Event:
    """One thing that happened: a verb with participants, for renderers and logs."""
    t: int
    kind: str
    data: Dict[str, Any]


class _SystemProcessor(esper.Processor):
    """Adapter: a plain `system(world, actions) -> events` function as an esper processor."""

    def __init__(self, fn: Callable[["World", List[Dict[str, Any]]], List[Event]]):
        self.fn = fn

    def process(self, world: "World", actions: List[Dict[str, Any]]) -> None:  # type: ignore[override]
        world._events.extend(self.fn(world, actions))


class World:
    def __init__(self, seed: int):
        import random

        self.rng = random.Random(seed)
        self.seed = seed
        self.t = 0
        self._name = f"mote-sim-{next(_ids)}"
        self._priority = itertools.count(10_000, -1)  # esper runs higher priority first
        self.names: Dict[int, str] = {}  # display name per entity (locale-independent key)
        self._rel: Dict[str, Dict[int, Set[int]]] = {}  # relation -> subject -> objects
        self._events: List[Event] = []
        self._use()

    # --- esper world bookkeeping ------------------------------------------------------
    def _use(self) -> None:
        esper.switch_world(self._name)

    def close(self) -> None:
        if self._name in esper.list_worlds():
            if esper.current_world == self._name:
                esper.switch_world("default")  # esper refuses to delete the active world
            esper.delete_world(self._name)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- entities/components ----------------------------------------------------------
    def spawn(self, name: str) -> int:
        self._use()
        eid = esper.create_entity()
        self.names[eid] = name
        return eid

    def eid(self, name: str) -> int:
        return next(k for k, v in self.names.items() if v == name)

    def name(self, eid: int) -> str:
        """The inverse of `eid` — a failure event has to say *who* holds the object it could not take."""
        return self.names[eid]

    def add(self, eid: int, comp: Any) -> None:
        self._use()
        esper.add_component(eid, comp)

    def get(self, eid: int, comp_type: type) -> Optional[Any]:
        self._use()
        return esper.try_component(eid, comp_type)

    def query(self, *comp_types: type) -> Iterator[Tuple]:
        """Iterate (eid, comp, ...) over entities holding every listed component, by eid."""
        self._use()
        for eid, comps in sorted(esper.get_components(*comp_types), key=lambda x: x[0]):
            yield (eid, *comps)

    # --- relationships ------------------------------------------------------------------
    def relate(self, subject: int, relation: str, obj: int) -> None:
        self._rel.setdefault(relation, {}).setdefault(subject, set()).add(obj)

    def unrelate(self, subject: int, relation: str, obj: Optional[int] = None) -> None:
        store = self._rel.get(relation, {})
        if subject in store:
            if obj is None:
                store[subject].clear()
            else:
                store[subject].discard(obj)

    def related(self, subject: int, relation: str) -> Set[int]:
        return set(self._rel.get(relation, {}).get(subject, ()))

    def one(self, subject: int, relation: str) -> Optional[int]:
        r = self.related(subject, relation)
        return next(iter(r)) if r else None

    def reverse(self, obj: int, relation: str) -> Set[int]:
        return {s for s, objs in self._rel.get(relation, {}).items() if obj in objs}

    def transitive(self, subject: int, relation: str) -> Set[int]:
        """Everything reachable from subject over relation (e.g. inside-of chains)."""
        seen: Set[int] = set()
        frontier = [subject]
        while frontier:
            nxt = self.related(frontier.pop(), relation) - seen
            seen |= nxt
            frontier.extend(nxt)
        return seen

    # --- dynamics -----------------------------------------------------------------------
    def add_system(self, fn: Callable[["World", List[Dict[str, Any]]], List[Event]]) -> None:
        self._use()
        esper.add_processor(_SystemProcessor(fn), priority=next(self._priority))

    def step(self, actions: List[Dict[str, Any]]) -> List[Event]:
        """Advance one tick: every processor sees the same action list, events accumulate."""
        self._use()
        self._events = []
        esper.process(self, actions)
        self.t += 1
        return list(self._events)

    # --- state ----------------------------------------------------------------------------
    def serialize(self) -> bytes:
        """Stable byte encoding of the full state (sorted keys throughout)."""
        self._use()
        comps: Dict[str, Dict[str, Any]] = {}
        for eid in sorted(self.names):
            for c in esper.components_for_entity(eid):
                comps.setdefault(type(c).__name__, {})[str(eid)] = asdict(c)
        state = {
            "schema": SCHEMA_VERSION,
            "t": self.t,
            "names": {str(k): v for k, v in sorted(self.names.items())},
            "components": {k: comps[k] for k in sorted(comps)},
            "relations": {rel: {str(s): sorted(o) for s, o in sorted(store.items()) if o}
                          for rel, store in sorted(self._rel.items())},
        }
        return json.dumps(state, sort_keys=True, ensure_ascii=False).encode("utf-8")
