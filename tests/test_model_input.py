"""Unit tests for the model input adapter and train-time quality filtering.

Offline: the backbone is built with weights=None, and images are synthetic.

The assertions that matter:
  * the default adapter feeds the backbone [-1, 1] (MobileNetV2's pretraining
    range), identical across the three channels,
  * the legacy adapter is still buildable, for paired comparisons,
  * a quality-failed image is DROPPED from training (it would otherwise train
    as a blank frame carrying a real label) but KEPT in evaluation, where
    predictions are aligned row-by-row with the split DataFrame.
"""

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from modeling.baseline_classifier import build_baseline  # noqa: E402
from modeling.train import (  # noqa: E402
    SUPPORTED_MONITORS,
    _build_callbacks,
    _build_dataset,
    _compile,
    _numpy_preprocess,
)

SIZE = (64, 64)


def _adapter_output(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    probe = tf.keras.Model(model.input, model.get_layer("channel_expand").output)
    return probe(batch).numpy()


def test_default_adapter_maps_unit_range_to_imagenet_range():
    model = build_baseline(input_size=SIZE, weights=None)
    batch = np.stack([np.zeros((*SIZE, 1)), np.ones((*SIZE, 1)),
                      np.full((*SIZE, 1), 0.5)]).astype(np.float32)
    out = _adapter_output(model, batch)
    assert out.shape == (3, *SIZE, 3)
    np.testing.assert_allclose(out[0], -1.0)
    np.testing.assert_allclose(out[1], 1.0)
    np.testing.assert_allclose(out[2], 0.0, atol=1e-6)
    # Same grayscale value in all three channels.
    np.testing.assert_allclose(out[..., 0], out[..., 2])


def test_default_adapter_has_no_trainable_input_weights():
    model = build_baseline(input_size=SIZE, weights=None)
    assert model.get_layer("channel_expand").trainable_weights == []


def test_legacy_conv1x1_adapter_still_builds():
    model = build_baseline(input_size=SIZE, weights=None, input_adapter="conv1x1")
    assert isinstance(model.get_layer("channel_expand"), tf.keras.layers.Conv2D)


def test_unknown_adapter_raises():
    with pytest.raises(ValueError, match="input_adapter"):
        build_baseline(input_size=SIZE, weights=None, input_adapter="rescale")


def test_unknown_monitor_raises_before_training(tmp_path):
    with pytest.raises(ValueError, match="monitor"):
        _build_callbacks(str(tmp_path), monitor="val_spec")


def test_compile_produces_every_supported_monitor_metric():
    model = build_baseline(input_size=SIZE, weights=None)
    _compile(model, 1e-3, "bce", 2.0, None)
    x = np.random.default_rng(0).random((8, *SIZE, 1)).astype(np.float32)
    y = np.array([0, 1] * 4, dtype=np.float32)
    history = model.fit(x, y, validation_data=(x, y), epochs=1, verbose=0)
    for monitor in SUPPORTED_MONITORS:
        assert monitor in history.history, f"{monitor} not produced by _compile"


def test_numpy_preprocess_flags_blank_image_as_failed():
    image, passed = _numpy_preprocess(np.zeros((512, 512), dtype=np.uint8), SIZE, b"x.png")
    assert passed is False
    assert image.shape == (*SIZE, 1)
    assert not image.any()


def _write_blank_png(path):
    tf.io.write_file(str(path), tf.io.encode_png(np.zeros((512, 512, 1), dtype=np.uint8)))


def test_training_dataset_drops_quality_failures(tmp_path):
    _write_blank_png(tmp_path / "blank.png")
    df = pd.DataFrame({"image_path": ["blank.png"], "label": [1]})
    ds = _build_dataset(df, str(tmp_path), "image_path", "label", SIZE, 4,
                        shuffle=True, cache=False, drop_failed_quality=True)
    assert sum(int(x.shape[0]) for x, _ in ds) == 0


def test_eval_dataset_keeps_quality_failures_for_row_alignment(tmp_path):
    _write_blank_png(tmp_path / "blank.png")
    df = pd.DataFrame({"image_path": ["blank.png"], "label": [1]})
    ds = _build_dataset(df, str(tmp_path), "image_path", "label", SIZE, 4,
                        shuffle=False, cache=False)
    batches = list(ds)
    assert len(batches) == 1
    images, labels = batches[0]
    assert images.shape == (1, *SIZE, 1)
    assert labels.numpy().tolist() == [1.0]
