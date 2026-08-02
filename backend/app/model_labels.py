"""Shared model-label contract (ISOT: 0=fake, 1=true)."""

DEFAULT_LABEL_MAP = {0: "Misinformation", 1: "Truth"}


def display_label(class_id: int, config) -> str:
    """Convert checkpoint metadata or a legacy ISOT class to an API label."""
    labels = getattr(config, "id2label", {}) or {}
    raw_label = labels.get(class_id, labels.get(str(class_id)))
    normalized = str(raw_label or "").strip().lower()
    if normalized in {"fake", "false", "misinformation", "label_0"}:
        return "Misinformation"
    if normalized in {"true", "real", "truth", "label_1"}:
        return "Truth"
    return DEFAULT_LABEL_MAP[class_id]
