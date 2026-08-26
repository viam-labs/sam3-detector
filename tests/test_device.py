"""GPU detection helpers — same behavior as sam2-detector."""

from models.common import (
    SAM3_CKPT_NAME,
    SAM3_MODEL_ID,
    GatedCheckpointError,
    checkpoint_search_dirs,
    find_bundled_checkpoint,
    is_gated_access_error,
    wrap_checkpoint_error,
)
from models.loader import video_predictor_supported


def test_tqdm_silence_allows_hf_hub_download():
    """tqdm stub must keep set_lock so Hub's lazy hf_hub_download import works."""
    import models.tqdm_silence  # noqa: F401
    from huggingface_hub import hf_hub_download
    import tqdm

    assert callable(hf_hub_download)
    assert hasattr(tqdm.tqdm, "set_lock")


def test_sam3_model_builder_imports_after_tqdm_silence():
    """Regression for the sam3-segments resource build ImportError."""
    import models.tqdm_silence  # noqa: F401
    from sam3.model_builder import build_sam3_image_model

    assert callable(build_sam3_image_model)
    assert video_predictor_supported("cuda") is True
    assert video_predictor_supported("cpu") is False
    assert video_predictor_supported("mps") is False


def test_model_id_is_public_sam3_not_sam2():
    assert SAM3_MODEL_ID == "facebook/sam3"
    assert SAM3_CKPT_NAME == "sam3.pt"


def test_checkpoint_search_includes_repo_and_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dirs = checkpoint_search_dirs()
    assert any(str(tmp_path) == d or str(tmp_path) in d for d in dirs)
    found = find_bundled_checkpoint()
    # A developer checkout may already have checkpoints/sam3.pt; cwd must not
    # be the only place we look, and we must not invent a file under tmp_path.
    if found is None:
        return
    assert found.endswith(f"checkpoints/{SAM3_CKPT_NAME}")
    assert str(tmp_path) not in found


def test_gated_repo_error_is_explained():
    class GatedRepoError(Exception):
        pass

    err = GatedRepoError("Cannot access gated repo facebook/sam3")
    assert is_gated_access_error(err)
    wrapped = wrap_checkpoint_error(err)
    assert isinstance(wrapped, GatedCheckpointError)
    assert "download_checkpoint.sh" in str(wrapped)
    assert "huggingface.co/facebook/sam3" in str(wrapped)


def test_unrelated_error_is_not_rewritten():
    err = RuntimeError("CUDA out of memory")
    assert not is_gated_access_error(err)
    assert wrap_checkpoint_error(err) is err
