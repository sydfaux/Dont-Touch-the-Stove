from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                beta_1 = group["betas"][0]
                beta_2 = group["betas"][1]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                ### TODO: Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-code in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
                ###
                ###       Refer to the default project handout for more details.
                ### YOUR CODE HERE
                if "m" not in state and "v" not in state:
                    if torch.cuda.is_available():
                        device = torch.device('cuda')
                    else:
                        device = torch.device('cpu')
                    state["m"]  = torch.zeros(p.size()).to(device)
                    state["v"] = torch.zeros(p.size()).to(device)
                    state["t"] = 0

                t = state["t"] + 1
                state["t"] = t
                m_t = beta_1 * state["m"] + (1 - beta_1) * grad
                v_t = beta_2 * state["v"] + (1 - beta_2) * (grad ** 2)
                state["m"] = m_t
                state["v"] = v_t
                alpha_t = alpha * math.sqrt(1 - (beta_2 ** t))/(1 - (beta_1 ** t))
                p.data = p.data  - (alpha_t * m_t/(torch.sqrt(v_t) + eps) + alpha * weight_decay * p.data)


        return loss
