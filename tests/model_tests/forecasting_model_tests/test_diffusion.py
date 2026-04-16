import pytest
import torch

from models.unet.unet import UNet
from models.forecasting.diffusion import DDPM
from models.model_utils.noise_scheduler import CosineScheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_noise_schedule() -> torch.Tensor:
    """Cosine alpha-bar schedule with T=5 steps, shape (6,)."""
    return CosineScheduler(T=5, s=0.008).schedule()


@pytest.fixture(scope="module")
def small_unet() -> UNet:
    """Lightweight UNet: 3 in-channels (u_0 + u_noisy + 1 coeff), 1 out-channel."""
    return UNet(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        final_filters=1,
        encoder_dropout=0.0,
        input_size=0,
    )


@pytest.fixture(scope="module")
def small_ddpm(small_unet: UNet, small_noise_schedule: torch.Tensor) -> DDPM:
    """DDPM wrapping the small UNet with a 5-step cosine schedule."""
    return DDPM(
        denoising_model=small_unet,
        noise_schedule=small_noise_schedule,
        optimiser="adam",
        learning_rate=1e-4,
    )


# ---------------------------------------------------------------------------
# Noise schedule tests
# ---------------------------------------------------------------------------

class TestNoiseSchedule:
    """Verify CosineScheduler produces a well-formed alpha-bar schedule."""

    def test_schedule_shape(self):
        """Schedule for T steps must have T+1 entries."""
        T = 10
        schedule = CosineScheduler(T=T, s=0.008).schedule()
        assert schedule.shape == (T + 1,)

    def test_schedule_starts_at_one(self):
        """First alpha-bar value (t=0) should be 1.0 by construction."""
        schedule = CosineScheduler(T=10, s=0.008).schedule()
        assert torch.isclose(schedule[0], torch.tensor(1.0), atol=1e-5)

    def test_schedule_ends_near_zero(self):
        """Last alpha-bar value (t=T) should be close to 0."""
        schedule = CosineScheduler(T=10, s=0.008).schedule()
        assert schedule[-1] < 0.01

    def test_schedule_is_monotone_decreasing(self):
        """Alpha-bar values must strictly decrease from t=0 to t=T."""
        schedule = CosineScheduler(T=10, s=0.008).schedule()
        assert (schedule[:-1] > schedule[1:]).all()


# ---------------------------------------------------------------------------
# UNet backbone tests
# ---------------------------------------------------------------------------

unet_params = [
    # (in_channels, out_channels, kernel_size, final_filters, encoder_dropout, input_size, B, H)
    (2, 4, 3, 1, 0.0, 0, 2, 32),
    (3, 4, 3, 1, 0.1, 0, 1, 64),
    (4, 8, 3, 2, 0.0, 0, 2, 32),
]


class TestUNetBackbone:
    """Inference and tensor-dimension tests for the UNet denoising backbone."""

    @pytest.mark.parametrize(
        "in_channels, out_channels, kernel_size, final_filters, encoder_dropout, input_size, B, H",
        unet_params,
    )
    def test_output_shape(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        final_filters: int,
        encoder_dropout: float,
        input_size: int,
        B: int,
        H: int,
    ):
        """UNet forward must return (B, final_filters, H, W) for a square input."""
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            final_filters=final_filters,
            encoder_dropout=encoder_dropout,
            input_size=input_size,
        )
        x = torch.randn(B, in_channels, H, H)
        t = torch.randint(0, 10, (B,))

        with torch.no_grad():
            out = model(x, t)

        assert out.shape == (B, final_filters, H, H), (
            f"Expected ({B}, {final_filters}, {H}, {H}), got {tuple(out.shape)}"
        )

    @pytest.mark.parametrize(
        "in_channels, out_channels, kernel_size, final_filters, encoder_dropout, input_size, B, H",
        unet_params,
    )
    def test_output_is_finite(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        final_filters: int,
        encoder_dropout: float,
        input_size: int,
        B: int,
        H: int,
    ):
        """UNet forward must not produce NaN or Inf values."""
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            final_filters=final_filters,
            encoder_dropout=encoder_dropout,
            input_size=input_size,
        )
        x = torch.randn(B, in_channels, H, H)
        t = torch.randint(0, 10, (B,))

        with torch.no_grad():
            out = model(x, t)

        assert torch.isfinite(out).all(), "UNet output contains NaN or Inf"

    def test_output_shape_invariant_to_timestep_value(self, small_unet: UNet):
        """Output spatial shape must not depend on the specific value of t."""
        B, H = 2, 32
        x = torch.randn(B, small_unet.in_channels, H, H)

        with torch.no_grad():
            out_t1 = small_unet(x, torch.full((B,), 1))
            out_t4 = small_unet(x, torch.full((B,), 4))

        assert out_t1.shape == out_t4.shape, (
            "Output shape changed between different timestep values"
        )


# ---------------------------------------------------------------------------
# DDPM inference tests
# ---------------------------------------------------------------------------

class TestDDPMInference:
    """Inference and tensor-dimension tests for the DDPM reverse diffusion loop."""

    def test_output_shape(self, small_ddpm: DDPM, small_unet: UNet):
        """DDPM.forward must return (B, final_filters, H, W).

        u_0 conditioning channels = unet.in_channels - unet.final_filters,
        because DDPM concatenates u_0 with u_noise_curr (shape final_filters)
        before passing to the UNet.
        """
        B, H = 2, 32
        cond_channels = small_unet.in_channels - small_unet.final_filters
        u_0 = torch.randn(B, cond_channels, H, H)

        with torch.no_grad():
            out = small_ddpm(u_0)

        expected = (B, small_unet.final_filters, H, H)
        assert out.shape == expected, f"Expected {expected}, got {tuple(out.shape)}"

    def test_output_is_finite(self, small_ddpm: DDPM, small_unet: UNet):
        """DDPM reverse diffusion must not produce NaN or Inf."""
        B, H = 2, 32
        cond_channels = small_unet.in_channels - small_unet.final_filters
        u_0 = torch.randn(B, cond_channels, H, H)

        with torch.no_grad():
            out = small_ddpm(u_0)

        assert torch.isfinite(out).all(), "DDPM output contains NaN or Inf"

    def test_output_shape_matches_input_spatial_dims(self, small_ddpm: DDPM, small_unet: UNet):
        """DDPM output H and W must match the conditioning input H and W."""
        B, H, W = 1, 32, 32
        cond_channels = small_unet.in_channels - small_unet.final_filters
        u_0 = torch.randn(B, cond_channels, H, W)

        with torch.no_grad():
            out = small_ddpm(u_0)

        assert out.shape[2] == H and out.shape[3] == W, (
            f"Spatial dims mismatch: input ({H},{W}), output ({out.shape[2]},{out.shape[3]})"
        )
