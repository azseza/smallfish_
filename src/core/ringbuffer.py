from collections import deque
from typing import TypeVar, Generic, Iterator, List, Optional, Callable

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Fixed-size circular buffer backed by collections.deque."""

    __slots__ = ("_buf",)

    def __init__(self, capacity: int):
        self._buf: deque[T] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._buf.maxlen  # type: ignore

    def append(self, item: T) -> None:
        self._buf.append(item)

    def extend(self, items) -> None:
        self._buf.extend(items)

    def clear(self) -> None:
        self._buf.clear()

    def get(self) -> List[T]:
        return list(self._buf)

    def last(self, n: int = 1) -> List[T]:
        """Return the last n items (most recent first)."""
        if n >= len(self._buf):
            return list(reversed(self._buf))
        return [self._buf[-1 - i] for i in range(n)]

    def peek(self) -> Optional[T]:
        """Return the most recent item without removing it."""
        if self._buf:
            return self._buf[-1]
        return None

    def oldest(self) -> Optional[T]:
        if self._buf:
            return self._buf[0]
        return None

    def filter(self, predicate: Callable[[T], bool]) -> List[T]:
        return [x for x in self._buf if predicate(x)]

    def __len__(self) -> int:
        return len(self._buf)

    def __bool__(self) -> bool:
        return len(self._buf) > 0

    def __iter__(self) -> Iterator[T]:
        return iter(self._buf)

    def __getitem__(self, idx):
        return self._buf[idx]
