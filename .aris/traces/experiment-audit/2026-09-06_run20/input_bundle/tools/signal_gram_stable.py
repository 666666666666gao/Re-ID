"""Numerically stabilize Signal's existing three-vector Gram volume."""
def signal_gram_volume_stable(language, video, audio):
    import torch

    with torch.autocast("cuda", enabled=False):
        language, video, audio = language.float(), video.float(), audio.float()
        batch_size1, batch_size2 = language.shape[0], video.shape[0]
        ll = torch.einsum("bi,bi->b", language, language).unsqueeze(1).expand(-1, batch_size2)
        lv = language @ video.T
        la = language @ audio.T
        vv = torch.einsum("bi,bi->b", video, video).unsqueeze(0).expand(batch_size1, -1)
        va = torch.einsum("bi,bi->b", video, audio).unsqueeze(0).expand(batch_size1, -1)
        aa = torch.einsum("bi,bi->b", audio, audio).unsqueeze(0).expand(batch_size1, -1)
        G = torch.stack([
            torch.stack([ll, lv, la], dim=-1),
            torch.stack([lv, vv, va], dim=-1),
            torch.stack([la, va, aa], dim=-1),
        ], dim=-2)
        gram_det = torch.det(G.float())
        # Signal's Triplet distance already uses this numerical floor before sqrt.
        # A saved RGBNT100 source batch also has a zero determinant in FP32.
        res = torch.sqrt(torch.abs(gram_det).clamp_min(1e-12))
        return res
