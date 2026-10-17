"""E5 preprocessing and numeric contracts without downloading models in CI."""

import numpy as np
import pytest

from atlasir.encoding import MODEL_REVISION, E5Encoder, mean_pool_normalize, resolved_encoder_config


@pytest.mark.parametrize(
    "config",
    [
        {"model_revision": "main"},
        {"query_prefix": ""},
        {"normalize": False},
        {"max_length": 256},
        {"batch_size": True},
        {"batch_size": 0},
        {"dtype": "float16"},
        {"seed": -1},
        {"device": "auto"},
        {"unknown": 1},
    ],
)
def test_configuration_is_explicit(config):
    with pytest.raises(ValueError):
        resolved_encoder_config(config)
    assert len(MODEL_REVISION) == 40


def test_masked_pooling_and_normalization():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor(
        [[[3.0, 0.0], [0.0, 4.0], [999.0, 999.0]], [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert np.allclose(mean_pool_normalize(hidden, mask).numpy(), [[0.6, 0.8], [0, 1]])
    with pytest.raises(ValueError, match="without attended"):
        mean_pool_normalize(hidden, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="zero-norm"):
        mean_pool_normalize(torch.zeros_like(hidden), mask)


def test_encoding_tracks_prefix_lengths_padding_and_truncation():
    torch = pytest.importorskip("torch")

    class Tokens(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, batch, **kwargs):
            self.calls.append((batch, kwargs))
            rows = [[101] + [len(word) % 10 + 1 for word in text.split()] + [102] for text in batch]
            if not kwargs.get("truncation"):
                return {"input_ids": rows}
            rows = [row[: kwargs["max_length"]] for row in rows]
            length = max(map(len, rows))
            return Tokens(
                input_ids=torch.tensor([r + [0] * (length - len(r)) for r in rows]),
                attention_mask=torch.tensor([[1] * len(r) + [0] * (length - len(r)) for r in rows]),
            )

    class Model:
        def __call__(self, input_ids, attention_mask):
            class Result:
                pass

            result = Result()
            result.last_hidden_state = input_ids[..., None].float().expand(-1, -1, 384)
            return result

    model = E5Encoder.__new__(E5Encoder)
    model.config = resolved_encoder_config({"batch_size": 2})
    model.tokenizer = Tokenizer()
    model.model = Model()
    texts = ["short sentence", "long " * 600]
    result = model.encode(["q1", "q2"], texts, kind="query")
    assert result.ids == ("q1", "q2")
    assert result.vectors.shape == (2, 384)
    assert result.vectors.dtype == np.float32 and not result.vectors.flags.writeable
    assert result.describe()["truncated_ids"] == ["q2"]
    assert model.tokenizer.calls[0][0][0] == "query: short sentence"
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1)
    repeat = model.encode(["q1", "q2"], texts, kind="query")
    assert np.array_equal(result.vectors, repeat.vectors)
    passage = model.encode(["d1"], ["short sentence"], kind="passage")
    assert model.tokenizer.calls[-2][0] == ["passage: short sentence"]
    assert passage.input_sha256 != result.input_sha256
    assert model.encode([], [], kind="query").vectors.shape == (0, 384)
    for ids, values in [(["a", "a"], ["a", "b"]), (["a"], [""]), (["a"], [])]:
        with pytest.raises(ValueError):
            model.encode(ids, values, kind="query")


def test_encoding_bundle_roundtrip_without_torch(tmp_path):
    import io
    from pathlib import Path

    from atlasir.artifacts import RunWriter, export_run, restore_run, verify_run

    source = Path(__file__).resolve().parents[1]
    inputs = (source / "configs/fixtures/e5-pilot.json").read_bytes()
    array = io.BytesIO()
    np.savez_compressed(array, queries=np.ones((1, 384), dtype=np.float32))
    with RunWriter(
        tmp_path,
        source,
        resolved_encoder_config(),
        None,
        "e02-preservation-test",
        encoder_payloads={"pilot-inputs.json": inputs},
    ) as writer:
        for name in ("results.json", "encoding.json", "model-files.json"):
            writer.write_json(name, {"scope": "schema fixture, not a real encoding"})
        writer.write_bytes("vectors.npz", array.getvalue())
        run = writer.seal()
    assert verify_run(run)["schema_version"] == 3
    bundle = export_run(run, tmp_path / "copy.bundle")
    restored = tmp_path / "restored"
    assert restore_run(bundle, restored)["run_id"] == "e02-preservation-test"
    assert (restored / "pilot-inputs.json").read_bytes() == inputs
    (restored / "vectors.npz").write_bytes(b"corruption")
    with pytest.raises(ValueError):
        verify_run(restored)


def test_day06_upload_bootstrap(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    source = Path(__file__).resolve().parents[1]
    package = tmp_path / "day06.zip"
    subprocess.run(
        [
            sys.executable,
            str(source / "scripts/prepare_kaggle_day06.py"),
            "--destination",
            str(package),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    uploaded = tmp_path / "uploaded"
    with zipfile.ZipFile(package) as archive:
        assert not any(name.endswith((".gz", ".tar", ".zip")) for name in archive.namelist())
        archive.extractall(uploaded)
    notebook = json.loads((source / "notebooks/kaggle-day06.ipynb").read_bytes())
    namespace = {"ATLASIR_INPUT_ROOT": uploaded, "ATLASIR_WORKING_ROOT": tmp_path / "work"}
    exec(compile("".join(notebook["cells"][1]["source"]), "day06-bootstrap", "exec"), namespace)
    assert (
        hashlib.sha256((namespace["INPUT"] / "source.tar.gz").read_bytes()).hexdigest()
        == namespace["manifest"]["sha256"]["source.tar.gz"]
    )
    assert (namespace["SOURCE"] / "configs/fixtures/e5-pilot.json").is_file()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "day06-cell", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
