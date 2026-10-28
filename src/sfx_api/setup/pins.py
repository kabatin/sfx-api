"""Everything ``setup`` installs, pinned. Sizes are static (bytes from the Hugging Face API,
2026-10-28) so ``--check`` needs no network."""

from __future__ import annotations

from dataclasses import dataclass

SA3_REPO = "https://github.com/Stability-AI/stable-audio-3"
SA3_COMMIT = "77e0398c26f7d22d801ec1bb036307e8b231f741"
SA3_PYTHON = "3.11"  # install.sh's own default
# install.sh's bundle name per component (``--download a,b``)
SA3_BUNDLES = {"sa3-medium": "medium", "sa3-small": "sm-sfx"}
# what install.py downloads per bundle into ``models/mlx/`` (it also fetches the encoders)
SA3_FILES = {
    "medium": {
        "dit_medium_f16.npz": 2_907_300_946,
        "same_l_decoder_f32.npz": 1_704_311_976,
        "same_l_encoder_f32.npz": 1_704_313_504,
    },
    "sm-sfx": {
        "dit_sm-sfx_f16.npz": 919_193_814,
        "same_s_decoder_f32.npz": 218_090_820,
        "same_s_encoder_f32.npz": 214_946_620,
    },
}
SA3_SHARED_FILES = {"t5gemma_f16.npz": 567_443_068}
# mlx, numpy, sentencepiece, huggingface_hub, soundfile, pillow (estimate)
SA3_ENV_BYTES = 200_000_000

MOSS_REPO = "https://github.com/OpenMOSS/MOSS-TTS"
MOSS_COMMIT = "934d6826b084c46a0d033402174d5f8ac4ed2519"
MOSS_PYTHON = "3.12"
# upstream moss_soundeffect_v2/pyproject.toml without the CUDA extra and gradio (README)
MOSS_PACKAGES = (
    "torch==2.9.1",
    "torchaudio==2.9.1",
    "numpy==1.26.4",
    "einops==0.8.2",
    "pillow==12.2.0",
    "tqdm==4.67.3",
    "safetensors==0.7.0",
    "transformers==4.57.1",
    "diffusers==0.37.1",
    "ftfy==6.3.1",
    "regex==2026.4.4",
    "soundfile==0.13.1",
    "descript-audiotools==0.7.2",
)
MOSS_MODEL_REPO = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
MOSS_MODEL_REVISION = "e35df4d82fbe87fcd5d14e5d100e349c0c3c076d"
MOSS_MODEL_BYTES = 11_230_170_282
# torch, transformers, diffusers, descript-audiotools and their dependencies (estimate)
MOSS_ENV_BYTES = 1_000_000_000
# the CLI that downloads the MOSS model, run with uvx so the engine's own environment stays
# exactly the README's pins
HF_CLI = "huggingface_hub==2.3.0"


@dataclass(frozen=True)
class Licence:
    name: str
    note: str


SA3_LICENCE = Licence(
    "Stability AI Community License",
    "free below US$1M annual revenue; commercial use requires registering with Stability AI "
    "(https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/main/LICENSE.md)",
)
MOSS_LICENCE = Licence("Apache-2.0", "code and weights")
