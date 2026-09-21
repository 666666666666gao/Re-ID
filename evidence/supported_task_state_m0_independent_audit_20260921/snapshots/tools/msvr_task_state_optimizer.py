"""AdaTask-inspired supported R/A states; fixed FP32 AdamW contract.

No data/model code. Full current+historical R and direct A must be unscaled
before step. Parameter gradients remain available for GradScaler's checks.
"""
import torch


class SupportedTaskAdamW(torch.optim.Optimizer):
    def __init__(self, parameters, role_parameters, *, split, lr, weight_decay):
        parameters = list(parameters)
        super().__init__(parameters, dict(lr=lr, weight_decay=weight_decay,
                                         betas=(0.9, 0.999), eps=1e-8))
        self.roles = list(role_parameters)
        self.role_ids = {id(p): i for i, p in enumerate(self.roles)}
        assert len(self.role_ids) == len(self.roles)
        assert set(self.role_ids).issubset({id(p) for p in parameters})
        assert all(p.dtype == torch.float32 for p in parameters)
        self.split = bool(split)

    @staticmethod
    def direction(state, key, gradient, beta1, beta2, eps):
        if key not in state:
            state[key] = dict(step=0, exp_avg=torch.zeros_like(gradient),
                              exp_avg_sq=torch.zeros_like(gradient))
        task = state[key]
        task['step'] += 1
        task['exp_avg'].lerp_(gradient, 1-beta1)
        task['exp_avg_sq'].mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)
        denominator = task['exp_avg_sq'].sqrt() / (1-beta2**task['step'])**0.5
        denominator.add_(eps)
        return (task['exp_avg'] / (1-beta1**task['step'])) / denominator

    @torch.no_grad()
    def step(self, *, rank_gradients, auxiliary_gradients, rank_observed):
        assert len(rank_gradients) == len(auxiliary_gradients) == len(self.roles)
        for p, r, a in zip(self.roles, rank_gradients, auxiliary_gradients, strict=True):
            assert r.shape == a.shape == p.shape
            assert r.dtype == a.dtype == p.dtype
            assert r.device == a.device == p.device
            assert torch.isfinite(r).all() and torch.isfinite(a).all()
            if not rank_observed:
                assert not bool(r.abs().sum())
        # Validate all gradients before mutating any parameter or task state.
        for group in self.param_groups:
            for p in group['params']:
                assert p.grad is not None and torch.isfinite(p.grad).all()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                state = self.state[p]
                if id(p) in self.role_ids:
                    i = self.role_ids[id(p)]
                    r, a = rank_gradients[i], auxiliary_gradients[i]
                    if self.split:
                        update = self.direction(state, 'auxiliary', a, beta1, beta2, group['eps'])
                        if rank_observed:
                            update = update + self.direction(state, 'rank', r, beta1, beta2, group['eps'])
                    else:
                        update = self.direction(state, 'shared', r+a, beta1, beta2, group['eps'])
                else:
                    update = self.direction(state, 'head', p.grad, beta1, beta2, group['eps'])
                p.mul_(1-group['lr']*group['weight_decay'])
                p.add_(update, alpha=-group['lr'])
