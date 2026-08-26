"""GPU detection helpers — same behavior as sam2-detector."""

from models.common import SAM3_CKPT_NAME, SAM3_MODEL_ID, checkpoint_search_dirs, find_bundled_checkpoint
from models.loader import video_predictor_supported


def test_video_predictor_only_on_cuda():
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
    assert find_bundled_checkpoint() is None
