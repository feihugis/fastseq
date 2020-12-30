# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimized Softmax to support larger input size"""

import torch
from torch import Tensor
from torch.nn import Module, Softmax
from torch._overrides import has_torch_function, handle_torch_function
from torch.nn.functional import softmax, _get_softmax_dim
from torch.nn import functional as F

from fairseq.modules.gelu import gelu
from fastseq.utils.api_decorator import replace

MAX_INPUT_SIZE = 2048 * 1024 * 1024

@replace(gelu)
def gelu_v2(x: torch.Tensor) -> torch.Tensor:
    print("++++++ GELU V2")
    return torch.nn.functional.gelu(x).type_as(x)

# @replace(Softmax)
# class SoftmaxV2(Module):

#     __constants__ = ['dim']

#     def __init__(self, dim=None):
#         super(SoftmaxV2, self).__init__()
#         self.dim = dim

#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         if not hasattr(self, 'dim'):
#             self.dim = None

#     def forward(self, input: Tensor) -> Tensor:
#         num_el = input.numel()
#         num_chunks = num_el // MAX_INPUT_SIZE + 1
#         stride = MAX_INPUT_SIZE // input.size()[1:].numel()
#         chunks = [F.softmax(
#             input[stride * i : stride * (i + 1), ], self.dim, _stacklevel=5)
#             for i in range(num_chunks)]
#         return torch.cat(chunks, dim=0)

#     def extra_repr(self):
#         return 'dim={dim}'.format(dim=self.dim)


@replace(softmax)
def softmax_v2(input, dim=None, _stacklevel=10, dtype=None):
    print("------")
    if not torch.jit.is_scripting():
        if type(input) is not Tensor and has_torch_function((input,)):
            return handle_torch_function(
                softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    if dim is None:
        dim = _get_softmax_dim('softmax', input.dim(), _stacklevel)
    num_el = input.numel()
    num_chunks = num_el // MAX_INPUT_SIZE + 1
    stride = MAX_INPUT_SIZE // input.size()[1:].numel()

    output = torch.zeros(input.shape, device=input.device, dtype=input.dtype)
    if dtype is None:
        # chunks = [
        #     input[stride * i : stride * (i + 1), ].softmax(dim)
        #     for i in range(num_chunks)]
        for i in range(num_chunks):
            output[stride * i : stride * (i + 1), ] = input[stride * i : stride * (i + 1), ].softmax(dim)
    else:
        # chunks = [
        #     input[stride * i : stride * (i + 1), ].softmax(dim, dtype=dtype)
        #     for i in range(num_chunks)]
        for i in range(num_chunks):
            output[stride * i : stride * (i + 1), ] = input[stride * i : stride * (i + 1), ].softmax(dim, dtype=dtype)
    return output
