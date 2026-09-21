"""Central settings. [REPORT] values where stated, [CHOICE] defaults otherwise."""
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    checkpoint: Path = Path("training/checkpoints/best.pt")
    upload_dir: Path = Path("storage/uploads")
    output_dir: Path = Path("storage/outputs")
    target_spacing: tuple = (1.0, 1.0, 1.0)  # [CHOICE]
    patch_size: tuple = (96, 96, 96)  # [CHOICE]
    top_k_slices: int = 6  # [REPORT] Fig 4.4 shows 6 slices
    threshold: float = 0.5
    quality_threshold: float = 1e-6  # [CHOICE] min tri-view score for Quality OK?
    min_lesion_voxels: int = 20  # [CHOICE] post-processing small-component filter
    llm_model: str = "claude-sonnet-5"
    llm_api_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
