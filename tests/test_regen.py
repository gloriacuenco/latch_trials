import torch

from hdev.regen import coarse_part, guided, regenerate


class Constant(torch.nn.Module):
    """Stand-in velocity field: a fixed velocity per site label, so the integrator can be checked exactly."""

    unknown = 2

    def forward(self, z, t, site, anatomy):
        return torch.ones_like(z) * site[:, None, None, None, None].float()


def test_keep_one_returns_the_source():
    z, noise = torch.randn(2, 4, 3, 3, 3), torch.randn(2, 4, 3, 3, 3)
    out = regenerate(Constant(), z, noise, torch.zeros(2, dtype=torch.long), torch.zeros(2, 10, 3, 3, 3), keep=1.0)
    assert torch.equal(out, z)


def test_noised_start_then_integrates_to_t_one():
    z, noise = torch.randn(2, 4, 3, 3, 3), torch.randn(2, 4, 3, 3, 3)
    site = torch.ones(2, dtype=torch.long)  # velocity 1 everywhere
    out = regenerate(Constant(), z, noise, site, torch.zeros(2, 10, 3, 3, 3), keep=0.4, steps=7)
    assert torch.allclose(out, 0.4 * z + 0.6 * noise + 0.6, atol=1e-5)


def test_guidance_extrapolates_from_the_unknown_site():
    z = torch.zeros(1, 4, 2, 2, 2)
    v = guided(Constant(), z, torch.zeros(1), torch.ones(1, dtype=torch.long), torch.zeros(1, 10, 2, 2, 2), 3.0)
    assert torch.allclose(v, torch.full_like(z, 2 + 3 * (1 - 2)))


def test_fine_scale_changes_only_the_coarse_part():
    torch.manual_seed(0)
    z, noise = torch.randn(1, 4, 16, 16, 16), torch.randn(1, 4, 16, 16, 16)
    site = torch.ones(1, dtype=torch.long)  # velocity 1 everywhere
    out = regenerate(Constant(), z, noise, site, torch.zeros(1, 10, 16, 16, 16), keep=0.0, steps=5, fine_scale=4)

    def detail(x):
        return (x - coarse_part(x, 4)).abs().mean()

    assert detail(out - z) < 0.2 * detail(noise)  # the change is smooth: the source's detail is kept
    assert (out - z).abs().mean() > 0.1  # and the coarse part was re-drawn
