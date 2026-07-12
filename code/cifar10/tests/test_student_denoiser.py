# Smoke tests for the latent-conditioned StudentDenoiser
# (code/torchcfm/models/student_denoiser.py). No pytest dependency required --
# run directly:
#   python code/cifar10/tests/test_student_denoiser.py
# or, if pytest is installed:
#   pytest code/cifar10/tests/test_student_denoiser.py

import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "torchcfm" / "models"))

import torch

from student_denoiser import StudentDenoiser, load_student

LATENT_DIMS = [64, 128, 256, 384, 512, 1024]


def test_forward_shapes():
    for dim in LATENT_DIMS:
        student = StudentDenoiser(latent_input_dim=dim, latent_embed_dim=256, hidden_channels=32, n_blocks=2)
        x0 = torch.randn(4, 3, 32, 32)
        latent = torch.randn(4, dim)
        output = student(x0, latent)
        assert output.shape == x0.shape, f"dim={dim}: expected {x0.shape}, got {output.shape}"


def test_forward_rejects_wrong_latent_dim():
    student = StudentDenoiser(latent_input_dim=128, latent_embed_dim=256, hidden_channels=32, n_blocks=2)
    x0 = torch.randn(4, 3, 32, 32)
    wrong_latent = torch.randn(4, 64)
    try:
        student(x0, wrong_latent)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched latent_input_dim")


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        for dim in [128, 512]:
            student = StudentDenoiser(latent_input_dim=dim, latent_embed_dim=256, hidden_channels=32, n_blocks=2)
            ckpt_path = Path(tmp) / f"student_{dim}.pt"
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "latent_dim": dim,
                    "latent_embed_dim": 256,
                    "dataset_size": 1000,
                    "student_hidden_channels": 32,
                    "student_n_blocks": 2,
                    "conditioning": "ae_latent",
                },
                ckpt_path,
            )
            loaded = load_student(str(ckpt_path), latent_dim=dim, device="cpu")
            x0 = torch.randn(2, 3, 32, 32)
            latent = torch.randn(2, dim)
            with torch.no_grad():
                out_original = student(x0, latent)
                out_loaded = loaded(x0, latent)
            assert torch.allclose(out_original, out_loaded), f"dim={dim}: loaded model output diverged"


def test_load_student_rejects_wrong_dim():
    with tempfile.TemporaryDirectory() as tmp:
        student = StudentDenoiser(latent_input_dim=128, latent_embed_dim=256, hidden_channels=32, n_blocks=2)
        ckpt_path = Path(tmp) / "student_128.pt"
        torch.save(
            {
                "model_state_dict": student.state_dict(),
                "latent_dim": 128,
                "latent_embed_dim": 256,
                "dataset_size": 1000,
                "student_hidden_channels": 32,
                "student_n_blocks": 2,
                "conditioning": "ae_latent",
            },
            ckpt_path,
        )
        try:
            load_student(str(ckpt_path), latent_dim=256, device="cpu")
        except ValueError:
            return
        raise AssertionError("expected ValueError for latent_dim mismatch")


def test_load_student_rejects_legacy_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "legacy_student.pt"
        torch.save(
            {
                "model_state_dict": {},
                "latent_dim": 128,
                "student_hidden_channels": 32,
                "student_n_blocks": 2,
                # no "conditioning" key -- mimics a pre-latent-conditioning checkpoint
            },
            ckpt_path,
        )
        try:
            load_student(str(ckpt_path), latent_dim=128, device="cpu")
        except ValueError:
            return
        raise AssertionError("expected ValueError for legacy (latent-free) checkpoint")


if __name__ == "__main__":
    tests = [
        test_forward_shapes,
        test_forward_rejects_wrong_latent_dim,
        test_checkpoint_roundtrip,
        test_load_student_rejects_wrong_dim,
        test_load_student_rejects_legacy_checkpoint,
    ]
    for test in tests:
        test()
        print(f"  [PASS] {test.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
