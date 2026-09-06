"""Keep Signal's existing Gram-volume calculation in FP32 during AMP."""
def signal_gram_volume_fp32(language, video, audio):
    import torch
    from utils.volume import volume_computation3

    with torch.autocast("cuda", enabled=False):
        return volume_computation3(language.float(), video.float(), audio.float())
