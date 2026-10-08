"""Corrected candidate for the level1 matrix-scalar multiplication sample."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matrix_scalar_mul_kernel(input_ptr, output_ptr, scalar, n_elements, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0)
    program_count = tl.num_programs(0)
    block_count = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(program_id, block_count, program_count):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, values * scalar, mask=mask)


class Model(nn.Module):
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        source = A.contiguous()
        output = torch.empty_like(source)
        if source.numel() == 0:
            return output
        block_size = 4096
        grid = (min(triton.cdiv(source.numel(), block_size), 256),)
        matrix_scalar_mul_kernel[grid](
            source, output, s, source.numel(), BLOCK_SIZE=block_size
        )
        return output
