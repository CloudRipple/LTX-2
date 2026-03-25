import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .primitives import StateDict
from .sd_ops import SDOps


class Registry(Protocol):
    """
    Protocol for managing state dictionaries in a registry.
    It is used to store state dictionaries and reuse them later without loading them again.
    Implementations must provide:
    - add: Add a state dictionary to the registry
    - pop: Remove a state dictionary from the registry
    - get: Retrieve a state dictionary from the registry
    - clear: Clear all state dictionaries from the registry
    """

    def add(self, paths: list[str], sd_ops: SDOps | None, state_dict: StateDict) -> str | None: ...

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None: ...

    def get(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None: ...

    def get_or_add(self, paths: list[str], sd_ops: SDOps | None, loader: Callable[[], StateDict]) -> StateDict: ...

    def clear(self) -> None: ...


class DummyRegistry(Registry):
    """
    Dummy registry that does not store state dictionaries.
    """

    def add(self, paths: list[str], sd_ops: SDOps | None, state_dict: StateDict) -> None:
        pass

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        pass

    def get(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        pass

    def get_or_add(self, paths: list[str], sd_ops: SDOps | None, loader: Callable[[], StateDict]) -> StateDict:
        return loader()

    def clear(self) -> None:
        pass


@dataclass
class StateDictRegistry(Registry):
    """
    Registry that stores state dictionaries in a dictionary.
    """

    _state_dicts: dict[str, StateDict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _loading_events: dict[str, threading.Event] = field(default_factory=dict)

    def _generate_id(self, paths: list[str], sd_ops: SDOps | None) -> str:
        m = hashlib.sha256()
        parts = sorted(str(Path(p).resolve()) for p in paths)
        if sd_ops is not None:
            parts.append(sd_ops.name)
        m.update("\0".join(parts).encode("utf-8"))
        return m.hexdigest()

    def add(self, paths: list[str], sd_ops: SDOps | None, state_dict: StateDict) -> str:
        sd_id = self._generate_id(paths, sd_ops)
        with self._lock:
            if sd_id in self._state_dicts:
                raise ValueError(f"State dict retrieved from {paths} with {sd_ops} already added, check with get first")
            self._state_dicts[sd_id] = state_dict
        return sd_id

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        with self._lock:
            return self._state_dicts.pop(self._generate_id(paths, sd_ops), None)

    def get(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        with self._lock:
            return self._state_dicts.get(self._generate_id(paths, sd_ops), None)

    def get_or_add(self, paths: list[str], sd_ops: SDOps | None, loader: Callable[[], StateDict]) -> StateDict:
        sd_id = self._generate_id(paths, sd_ops)

        while True:
            with self._lock:
                state_dict = self._state_dicts.get(sd_id)
                if state_dict is not None:
                    return state_dict

                loading_event = self._loading_events.get(sd_id)
                if loading_event is None:
                    loading_event = threading.Event()
                    self._loading_events[sd_id] = loading_event
                    break

            loading_event.wait()

        try:
            state_dict = loader()
        except Exception:
            with self._lock:
                self._loading_events.pop(sd_id, None)
                loading_event.set()
            raise

        with self._lock:
            cached_state_dict = self._state_dicts.get(sd_id)
            if cached_state_dict is None:
                self._state_dicts[sd_id] = state_dict
                cached_state_dict = state_dict
            self._loading_events.pop(sd_id, None)
            loading_event.set()
            return cached_state_dict

    def clear(self) -> None:
        with self._lock:
            self._state_dicts.clear()
            self._loading_events.clear()
