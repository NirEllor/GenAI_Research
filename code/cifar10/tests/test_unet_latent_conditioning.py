# Smoke tests for deterministic latent conditioning in
# code/torchcfm/models/unet/unet_resnetVAE.py's UNetModelWrapper.
# No pytest dependency required -- run directly:
#   python code/cifar10/tests/test_unet_latent_conditioning.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "torchcfm" / "models" / "unet"))

import torch
from unet_resnetVAE import UNetModelWrapper

LATENT_DIMS = [64, 128, 256, 384, 512, 1024]


def test_forward_shapes_and_plain_tensor_output():
    for dim in LATENT_DIMS:
        model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_channels=64,          # smallest value safe against num_head_channels=64 divisibility assert
            num_res_blocks=1,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0,
            num_latents=dim,
        )
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        t = torch.rand(2)
        y = torch.randn(2, dim)
        with torch.no_grad():
            out = model(t, x, y=y)
        assert isinstance(out, torch.Tensor), f"dim={dim}: expected plain Tensor, got {type(out)}"
        assert out.shape == (2, 3, 32, 32), f"dim={dim}: expected (2,3,32,32), got {out.shape}"


def test_no_latent_dim_kwarg_accepted():
    # constructor must reject the removed latent_dim kwarg outright (TypeError),
    # not silently ignore it -- guards against a stale call site left un-migrated.
    try:
        UNetModelWrapper(
            dim=(3, 32, 32), num_channels=64, num_res_blocks=1,
            channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
            attention_resolutions="16", dropout=0, num_latents=64, latent_dim=64,
        )
    except TypeError:
        return
    raise AssertionError("expected TypeError: latent_dim should no longer be an accepted kwarg")


if __name__ == "__main__":
    tests = [test_forward_shapes_and_plain_tensor_output, test_no_latent_dim_kwarg_accepted]
    for test in tests:
        test()
        print(f"  [PASS] {test.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
