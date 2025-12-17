import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler


# profiler: tiempo de cpu, uso de memoria, stack trace y puede agrupar por funciones etiquetadas

# primera versión (ineficiente)


# 1 sola capa lineal
class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        # convierte threshold a scalar de python
        # mueve mask de gpu a cpu
        # convierte el resultado en tensor
        # lo devuelve a la gpu
        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            hi_idx = torch.from_numpy(hi_idx).cuda()

        return out, hi_idx


model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((350, 350, 350), dtype=torch.double).cuda()


model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cpu_time_total", row_limit=5
    )
)

# segunda versión (float64 vs float32)

model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand(
    (350, 350, 350), dtype=torch.float
).cuda()  # usamos float32 en lugar de float64 para usar menos memoria y es más rápido en gpu

model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cpu_time_total", row_limit=5
    )
)

# tercera versión (correcta y eficiente)


class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        # todo es pytorch
        # todo se ejecuta en la gpu
        # no hay cpu ni conversión a numpy
        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)

        return out, hi_idx


model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((350, 350, 350), dtype=torch.float).cuda()

model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cpu_time_total", row_limit=5
    )
)
