from types import SimpleNamespace

from app.model_labels import display_label


def test_legacy_isot_label_mapping():
    config = SimpleNamespace(id2label={})
    assert display_label(0, config) == "Misinformation"
    assert display_label(1, config) == "Truth"


def test_checkpoint_label_metadata_is_honored():
    config = SimpleNamespace(id2label={0: "Misinformation", 1: "Truth"})
    assert display_label(0, config) == "Misinformation"
    assert display_label(1, config) == "Truth"
