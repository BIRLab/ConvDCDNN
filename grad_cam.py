import torch


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, gap_axis=2, sum_axis=1):
        self.model = model.eval()
        self.activation = None
        self.handle = target_layer.register_forward_hook(self.save_activation)
        self.gap_axis = gap_axis
        self.sum_axis = sum_axis

    def __del__(self):
        self.handle.remove()

    def save_activation(self, _module, input, _output):
        self.activation = input[0]

    def __call__(self, target: torch.Tensor) -> torch.Tensor:
        alpha = torch.autograd.grad(target, self.activation, retain_graph=True)[0].mean(self.gap_axis, keepdim=True)
        return torch.relu(torch.sum(alpha * self.activation, self.sum_axis))


__all__ = ['GradCAM']
