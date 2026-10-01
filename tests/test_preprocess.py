import numpy as np

from hdev.preprocess import Placement, cut_fraction, from_grid, head_bbox, place, place_on_brain, resample, to_grid


def sphere(shape=(60, 70, 50), centre=(30, 35, 22), radius=15):
    idx = np.indices(shape)
    d2 = sum((i - c) ** 2 for i, c in zip(idx, centre))
    return (d2 <= radius**2).astype(np.float32)


def test_head_bbox_ignores_small_specks():
    x = sphere()
    x[0, 0, 0] = 1.0  # isolated speck
    assert head_bbox(x) == ((15, 46), (20, 51), (7, 38))


def test_grid_round_trip_is_exact_inside_the_grid():
    rng = np.random.default_rng(0)
    x = rng.random((60, 70, 50)).astype(np.float32)
    for p in [Placement((32, 48, 64), (10, -5, -7)), Placement((80, 80, 80), (-10, -5, -15))]:
        g = to_grid(x, p)
        assert g.shape == p.grid
        back = from_grid(g, p, x.shape, fill=x)
        np.testing.assert_array_equal(back, x)


def test_place_keeps_the_top_of_the_head_and_cuts_the_neck_first():
    x = sphere()
    p = place(head_bbox(x), grid=(48, 48, 24), top_margin=2)
    g = to_grid(x, p)
    # top of the sphere (z = 37) sits 2 voxels below the grid top
    assert g[:, :, -3].sum() > 0 and g[:, :, -2:].sum() == 0
    assert 0 < cut_fraction(x, p) < 0.5


def test_resample_round_trip_is_close():
    x = sphere(radius=18)
    up = resample(x, (90, 105, 75))
    back = resample(up, x.shape)
    assert np.abs(back - x).mean() < 0.02


def test_place_on_brain_centres_the_grid():
    p = place_on_brain(((15, 46), (20, 51), (7, 38)), grid=(48, 48, 48))
    assert p.offset == (6, 11, -2)
