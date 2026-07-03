from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class CacheSpec:
    artifact: str
    schema_version: int
    semantic_params: Mapping[str, Any] = field(default_factory=dict)
    producer_functions: Sequence[Callable] = field(default_factory=tuple)
    producer_constants: Mapping[str, Any] = field(default_factory=dict)
    upstream_keys: Mapping[str, str] = field(default_factory=dict)
