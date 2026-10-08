import torch

from hdev.vae import encode


def test_encode_uses_mean_even_when_stage2_samples():
    # Real MONAI API contract, tiny network; no pretrained weights or GPU required.
    from monai.networks.nets import AutoencoderKL
    model = AutoencoderKL(spatial_dims=3, in_channels=1, out_channels=1,
                          channels=(8, 8), num_res_blocks=1, latent_channels=4,
                          norm_num_groups=8, attention_levels=(False, False),
                          with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False).eval()
    x = torch.randn(1, 1, 8, 8, 8)
    with torch.no_grad():
        expected, _ = model.encode(x)
        torch.manual_seed(1)
        a = encode(model, x)
        torch.manual_seed(2)
        b = encode(model, x)
        sampled = model.encode_stage_2_inputs(x)
    torch.testing.assert_close(a, expected, rtol=0, atol=0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, sampled)



def test_diagnostic_selects_only_qc_passing_validation_subjects():
    import pandas as pd
    from scripts.e0_encoding_check import select_subjects
    meta = pd.DataFrame([
        {"subject_id": "A_train", "site": "A", "split": "train", "qc_exclude": False},
        {"subject_id": "A_test", "site": "A", "split": "test", "qc_exclude": False},
        {"subject_id": "A_bad", "site": "A", "split": "val", "qc_exclude": True},
        {"subject_id": "A_good", "site": "A", "split": "val", "qc_exclude": False},
        {"subject_id": "B_good", "site": "B", "split": "val", "qc_exclude": False},
    ])
    assert select_subjects(meta, 3).subject_id.tolist() == ["A_good", "B_good"]
