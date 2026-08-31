"""Source adapter registry."""

from __future__ import annotations

from ucc.logging_utils import get_logger
from ucc.sources.base import SourceAdapter

log = get_logger("sources")

# Import order == default priority order (lower = processed first).
_ADAPTER_CLASSES: dict[str, str] = {
    "common_pile_stackv2": "ucc.sources.common_pile:CommonPileStackV2Adapter",
    "common_pile_stackv2_edu": "ucc.sources.common_pile:CommonPileStackV2EduAdapter",
    "starcoder2data_extras": "ucc.sources.starcoder2_extras:StarCoder2ExtrasAdapter",
    "starcoderdata": "ucc.sources.starcoderdata:StarCoderDataAdapter",
    "the_stack_v2": "ucc.sources.the_stack_v2:TheStackV2Adapter",
}


def _load_class(spec: str):
    module_name, _, class_name = spec.partition(":")
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def build_adapters(cfg) -> dict[str, SourceAdapter]:
    """Instantiate every enabled adapter, in config order (priority)."""
    adapters: dict[str, SourceAdapter] = {}
    priority = 0
    for name, source_cfg in (cfg.get("sources") or {}).items():
        if not (source_cfg or {}).get("enabled", False):
            log.info("source %s disabled in config — skipping", name)
            continue
        if name not in _ADAPTER_CLASSES:
            log.warning("unknown source '%s' in config — skipping", name)
            continue
        cls = _load_class(_ADAPTER_CLASSES[name])
        adapters[name] = cls(name=name, cfg=cfg, priority=priority)
        priority += 1
    return adapters
