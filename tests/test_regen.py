import torch

from hdev.regen import guided, regenerate


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
