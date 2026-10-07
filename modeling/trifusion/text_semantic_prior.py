"""One frozen text-package intervention at the semantic role-read boundary."""
import torch
from torch import nn
import torch.nn.functional as F

from .role_input_detach import DetachedSemanticTriFusion


PERSON_TOKENS = (49406, 320, 1125, 539, 320, 343, 343, 343, 343, 2533, 269, 49407)
VEHICLE_TOKENS = (49406, 320, 1125, 539, 320, 343, 343, 343, 343, 5299, 269, 49407)


def text_package_state(public_state, package):
    """Return the pretraining package or the single registered random package.

    Random filling uses its own generator, independently of module construction
    and of the training/augmentation RNG. The temporary vocabulary is not a
    persistent model component.
    """
    assert package in ('pretrained', 'random')
    names = [name for name in public_state if name.startswith('transformer.resblocks.')]
    names += ['token_embedding.weight', 'positional_embedding',
              'ln_final.weight', 'ln_final.bias', 'text_projection']
    assert len(names) == 149 and len(set(names)) == 149
    if package == 'pretrained':
        return {name: public_state[name].detach().cpu().float() for name in names}
    generator = torch.Generator(device='cpu').manual_seed(42)
    state = {}

    def normal(name, std):
        value = torch.empty(public_state[name].shape, dtype=torch.float32, device='cpu')
        state[name] = value.normal_(std=std, generator=generator)

    normal('token_embedding.weight', 0.02)
    normal('positional_embedding', 0.01)
    for index in range(12):
        prefix = f'transformer.resblocks.{index}.'
        normal(prefix+'attn.in_proj_weight', 512 ** -0.5)
        normal(prefix+'attn.out_proj.weight', 512 ** -0.5 * 24 ** -0.5)
        normal(prefix+'mlp.c_fc.weight', 1024 ** -0.5)
        normal(prefix+'mlp.c_proj.weight', 512 ** -0.5 * 24 ** -0.5)
        for name in ('attn.in_proj_bias', 'attn.out_proj.bias', 'mlp.c_fc.bias', 'mlp.c_proj.bias',
                     'ln_1.bias', 'ln_2.bias'):
            state[prefix+name] = torch.zeros_like(public_state[prefix+name], device='cpu', dtype=torch.float32)
        for name in ('ln_1.weight', 'ln_2.weight'):
            state[prefix+name] = torch.ones_like(public_state[prefix+name], device='cpu', dtype=torch.float32)
    state['ln_final.weight'] = torch.ones(512, device='cpu', dtype=torch.float32)
    state['ln_final.bias'] = torch.zeros(512, device='cpu', dtype=torch.float32)
    normal('text_projection', 512 ** -0.5)
    assert set(state) == set(names)
    return state


class FrozenTextPackage(nn.Module):
    """Explicit ordinary CLIP blocks; frozen weights retain input autograd."""
    def __init__(self, clip_module, state, token_ids):
        super().__init__()
        assert tuple(token_ids) in (PERSON_TOKENS, VEHICLE_TOKENS)
        # Author constructors perform their own random initialization. These
        # draws must not affect the declared package filling or training RNG.
        with torch.random.fork_rng(devices=[]):
            self.blocks = nn.ModuleList([
                clip_module.ResidualAttentionBlock(d_model=512, n_head=8, pattern=None)
                for _ in range(12)
            ])
            self.ln_final = clip_module.LayerNorm(512)
        for index, block in enumerate(self.blocks):
            prefix = f'transformer.resblocks.{index}.'
            block.load_state_dict({name[len(prefix):]: value for name, value in state.items()
                                   if name.startswith(prefix)}, strict=True)
        self.ln_final.load_state_dict({name: state['ln_final.'+name]
                                      for name in ('weight', 'bias')}, strict=True)
        self.text_projection = nn.Parameter(state['text_projection'].clone())
        self.register_buffer('positional_embedding', state['positional_embedding'].clone())
        self.register_buffer('template_embedding', state['token_embedding.weight'][list(token_ids)].clone())
        assert self.positional_embedding.shape == (77, 512)
        assert self.template_embedding.shape == (12, 512)
        self.float().requires_grad_(False)
        self.eval()
        assert len(self.state_dict()) == 149
        assert sum(value.numel() for value in self.state_dict().values()) == 38137344

    def train(self, mode=True):
        return super().train(False)

    def encode_embeddings(self, embeddings):
        """Also admits full77 engineering reference; EOT is always index11."""
        length = embeddings.shape[1]
        assert length in (12, 77) and embeddings.shape[2] == 512
        assert embeddings.dtype == torch.float32
        mask = torch.full((length, length), float('-inf'), device=embeddings.device).triu_(1)
        x = (embeddings + self.positional_embedding[:length]).permute(1, 0, 2)
        for block in self.blocks:
            block.attn_mask = mask
            x = block.forward_ori(x)
        x = self.ln_final(x.permute(1, 0, 2))
        return x[:, 11] @ self.text_projection

    def forward(self, deltas):
        assert deltas.ndim == 3 and deltas.shape[1:] == (4, 512)
        template = self.template_embedding[None].expand(deltas.shape[0], -1, -1)
        embeddings = torch.cat((template[:, :5], template[:, 5:9] + deltas, template[:, 9:]), dim=1)
        return self.encode_embeddings(embeddings)


class TextContextConditioner(nn.Module):
    def __init__(self, frozen_text):
        super().__init__()
        self.text = frozen_text
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(42)
            self.psi = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 2048))
            self.output = nn.Linear(512, 512, bias=False)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) == 592000
        assert sum(p.requires_grad for p in self.parameters()) == 5

    def forward(self, shared_global, original_context):
        assert shared_global.ndim == 2 and shared_global.shape[1] == 1536
        assert original_context.shape == (shared_global.shape[0], 512)
        with torch.autocast(device_type=shared_global.device.type, enabled=False):
            images = F.normalize(shared_global.detach().float().reshape(-1, 512), dim=1)
            deltas = self.psi(images).reshape(-1, 4, 512)
            text = F.normalize(self.text(deltas), dim=1).reshape(-1, 3, 512)
            context = F.normalize(text.mean(dim=1), dim=1)
            return F.normalize(original_context.detach().float() + self.output(context), dim=1)


class TextContextTriFusion(DetachedSemanticTriFusion):
    """The builder attaches text_conditioner after constructing common state."""
    def role_evidence(self, batch, stages, context, shared_global):
        detached_global = shared_global.detach()
        augmented = self.text_conditioner(detached_global, context.detach())
        return self.roles(stages.detach(), augmented, detached_global)
