import torch


class ADCL(torch.optim.Optimizer):
    """Neural Network Optimizer Based on Neural Dynamic"""

    def __init__(self, params, lr=1e-3, activate_fn=None, vlr_clamp=1.0, weight_decay=0.0, trapezoidal=False, omicron=1e-3):
        super().__init__(params, defaults={
            'lr': lr,
            'activate_fn': activate_fn,
            'vlr_clamp': vlr_clamp,
            'weight_decay': weight_decay,
            'trapezoidal': trapezoidal,
            'omicron': omicron
        })

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError('NeuralDynamic optimizer need to pass loss through closure, please reference: "https://pytorch.org/docs/stable/optim.html#optimizer-step-closure".')

        with torch.enable_grad():
            loss = closure()

        for j, group in enumerate(self.param_groups):
            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    # calculate varying learning rate
                    if group['activate_fn'] is None:
                        vlr = group['lr'] * loss / (torch.sum(torch.square(p.grad)) + group['omicron'])
                    else:
                        vlr = group['lr'] * group['activate_fn'](loss) / (torch.sum(torch.square(p.grad)) + group['omicron'])

                    # learning rate clamp
                    if vlr > group['vlr_clamp']:
                        vlr = group['vlr_clamp']

                    # l2 regularization
                    if group['weight_decay'] > 0:
                        p.grad = p.grad.add(p, alpha=group['weight_decay'])

                    # optimization step
                    current_step = vlr * p.grad
                    if 'last_step' in self.state[p]:
                        p.data -= 0.5 * (current_step + self.state[p]['last_step'])
                    else:
                        p.data -= current_step

                    # trapezoidal integration
                    if group['trapezoidal']:
                        self.state[p]['last_step'] = current_step

        return loss


__all__ = ['ADCL']
