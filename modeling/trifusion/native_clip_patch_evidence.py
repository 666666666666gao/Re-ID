"""Auxiliary dense readings without replacing the CLIP global forward."""
import torch
import torch.nn.functional as F


def query_self_attention(attention, normalized_input):
    """QQ attention and original V/output projections; no residual or FFN.

    This standard attention calculation follows ClearCLIP's dense-reading idea.
    The caller retains the original global path and controls gradient ownership.
    """
    length, batch, width = normalized_input.shape
    heads = attention.num_heads
    assert width == attention.embed_dim and width % heads == 0
    assert attention.in_proj_weight.shape == (3 * width, width)
    head_width = width // heads
    q_weight, _, v_weight = attention.in_proj_weight.chunk(3)
    q_bias, _, v_bias = attention.in_proj_bias.chunk(3)
    q = F.linear(normalized_input.float(), q_weight.float(), q_bias.float())
    v = F.linear(normalized_input.float(), v_weight.float(), v_bias.float())
    q = q.reshape(length, batch, heads, head_width).permute(1, 2, 0, 3)
    v = v.reshape(length, batch, heads, head_width).permute(1, 2, 0, 3)
    weights = (q @ q.transpose(-1, -2) * head_width ** -.5).softmax(-1)
    output = (weights @ v).permute(2, 0, 1, 3).reshape(length, batch, width)
    return F.linear(output, attention.out_proj.weight.float(), attention.out_proj.bias.float())


def project_attention_patches(visual, attention_output):
    """Project actual auxiliary tokens in the same pretrained visual basis."""
    assert attention_output.shape[-1] == visual.proj.shape[0]
    tokens = visual.ln_post(attention_output.permute(1, 0, 2).float()) @ visual.proj.float()
    return tokens[:, 1:]
