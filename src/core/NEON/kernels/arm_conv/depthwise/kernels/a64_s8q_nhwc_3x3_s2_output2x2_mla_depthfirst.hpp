/*
 * Copyright (c) 2021-2024 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "utils.hpp"
#include "src/core/NEON/kernels/arm_conv/depthwise/interleaves/list.hpp"

#include <cstdint>

#pragma once

#if defined(__aarch64__)

namespace arm_conv {
namespace depthwise {

void a64_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst_impl(
  const unsigned int,
  const int8_t *const *const,
  const int8_t *const,
  const int32_t *const,
  const arm_gemm::Requantize32 &,
  const int32_t *const,
  const int32_t *const,
  int8_t *const *const
);


class a64_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst : public DepthwiseDepthfirstStrategy<int8_t, int8_t, int8_t, int32_t>
{
  using Parent = DepthwiseDepthfirstStrategy<int8_t, int8_t, int8_t, int32_t>;

  public:
  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 2;
  constexpr static unsigned int stride_cols = 2;

  a64_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst(const CPUInfo *) : Parent(2, 2, 3, 3, 2, 2) {}

  arm_gemm::VLType get_vl_type(void) const override { return arm_gemm::VLType::None; }

  Parent::KernelType kernel = a64_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
  unsigned int get_accumulator_depth_vl(void) const override { return 2; }
};

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__)
