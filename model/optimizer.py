# Based on https://github.com/pytorch/pytorch/pull/3740
# AdamW, RAdam
import math
import torch
# RAdam
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from torch import Tensor
LossClosure = Callable[[], float]
Betas2 = Tuple[float, float]
OptFloat = Optional[float]
OptLossClosure = Optional[LossClosure]
Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]


class AdamW(torch.optim.Optimizer):
    r"""Implements AdamW algorithm.
    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class RAdam(torch.optim.Optimizer):
    r"""Implements RAdam optimization algorithm.
    It has been proposed in `On the Variance of the Adaptive Learning
    Rate and Beyond`__.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.RAdam(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1908.03265
    Note:
        Reference code: https://github.com/LiyuanLucasLiu/RAdam
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if 'betas' in param and (
                    param['betas'][0] != betas[0]
                    or param['betas'][1] != betas[1]
                ):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    msg = (
                        'RAdam does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (
                        1 - beta2_t
                    )
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = (
                            lr
                            * math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            )
                            / (1 - beta1 ** state['step'])
                        )
                    else:
                        step_size = lr / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if weight_decay != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-weight_decay * lr)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size)

                p.data.copy_(p_data_fp32)

        return loss