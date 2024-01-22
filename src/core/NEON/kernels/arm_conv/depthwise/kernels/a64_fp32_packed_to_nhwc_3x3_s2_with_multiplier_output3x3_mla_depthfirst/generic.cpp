/*
 * Copyright (c) 2021, 2023 Arm Limited.
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

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__)

namespace arm_conv {
namespace depthwise {

void a64_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst_impl(
  const float *const *const inptrs,
  float *const *const outptrs,
  const void *params,
  const unsigned int n_output_channels,
  const float activation_min,
  const float activation_max
)
{
  const float minmax_vals[2] = { activation_min, activation_max };

  __asm__ __volatile__(
    "ld1r { v27.4s }, [%x[clamps]]\n"
    "ldr x21, [%x[inptrs], #0x0]\n"
    "lsr x22, %x[channel_multiplier], #0x2\n"
    "add x20, %x[clamps], #0x4\n"
    "ldr q0, [x21, #0x0]\n"
    "ldr q1, [x21, #0x10]\n"
    "mov x21, #0x0\n"
    "mov x14, #0x0\n"
    "ld1r { v26.4s }, [x20]\n"
    "ldr x20, [%x[inptrs], #0x8]\n"
    "ldr q2, [x20, #0x0]\n"
    "ldr q3, [x20, #0x10]\n"
    "ldr x20, [%x[inptrs], #0x10]\n"
    "ldr q4, [x20, #0x0]\n"
    "ldr q5, [x20, #0x10]\n"
    "ldr x20, [%x[inptrs], #0x18]\n"
    "ldr q6, [x20, #0x0]\n"
    "ldr q7, [x20, #0x10]\n"
    "ldr x20, [%x[inptrs], #0x20]\n"
    "ldr q8, [x20, #0x0]\n"
    "ldr q9, [x20, #0x10]\n"
    "ldr x20, [%x[inptrs], #0x28]\n"
    "ldr q10, [x20, #0x0]\n"
    "ldr q11, [x20, #0x10]\n"
    "ldr x20, [%x[inptrs], #0x30]\n"
    "ldr q12, [x20, #0x0]\n"
    "ldr q13, [x20, #0x10]\n"
    "ldp x13, x12, [%x[outptrs], #0x0]\n"
    "ldp x11, x10, [%x[outptrs], #0x10]\n"
    "ldp x9, x28, [%x[outptrs], #0x20]\n"
    "ldp x27, x26, [%x[outptrs], #0x30]\n"
    "ldr x25, [%x[outptrs], #0x40]\n"
    "cbz x22, 3f\n"
    "ldr q14, [%x[params], #0x0]\n"
    "ldr q31, [%x[params], #0x10]\n"
    "subs x22, x22, #0x1\n"
    "mov v15.16b, v14.16b\n"
    "ldr q30, [%x[params], #0x20]\n"
    "ldr q29, [%x[params], #0x30]\n"
    "mov v16.16b, v14.16b\n"
    "mov v17.16b, v14.16b\n"
    "mov v18.16b, v14.16b\n"
    "mov v19.16b, v14.16b\n"
    "add %x[params], %x[params], #0x40\n"
    "mov v20.16b, v14.16b\n"
    "mov v21.16b, v14.16b\n"
    "mov v22.16b, v14.16b\n"
    "beq 2f\n"
    "1:"  // Output channel complete vector loop
    "fmla v14.4s, v31.4s, v0.s[0]\n"
    "fmla v15.4s, v31.4s, v0.s[2]\n"
    "subs x22, x22, #0x1\n"
    "add x21, x21, #0x4\n"
    "fmla v16.4s, v31.4s, v1.s[0]\n"
    "fmla v17.4s, v31.4s, v4.s[0]\n"
    "fmla v18.4s, v31.4s, v4.s[2]\n"
    "fmla v19.4s, v31.4s, v5.s[0]\n"
    "fmla v20.4s, v31.4s, v8.s[0]\n"
    "fmla v21.4s, v31.4s, v8.s[2]\n"
    "fmla v22.4s, v31.4s, v9.s[0]\n"
    "ldr q25, [%x[params], #0x0]\n"
    "fmla v14.4s, v30.4s, v0.s[1]\n"
    "fmla v15.4s, v30.4s, v0.s[3]\n"
    "fmla v16.4s, v30.4s, v1.s[1]\n"
    "fmla v17.4s, v30.4s, v4.s[1]\n"
    "fmla v18.4s, v30.4s, v4.s[3]\n"
    "fmla v19.4s, v30.4s, v5.s[1]\n"
    "fmla v20.4s, v30.4s, v8.s[1]\n"
    "fmla v21.4s, v30.4s, v8.s[3]\n"
    "fmla v22.4s, v30.4s, v9.s[1]\n"
    "ldr q24, [%x[params], #0x10]\n"
    "fmla v14.4s, v29.4s, v0.s[2]\n"
    "fmla v15.4s, v29.4s, v1.s[0]\n"
    "fmla v16.4s, v29.4s, v1.s[2]\n"
    "fmla v17.4s, v29.4s, v4.s[2]\n"
    "fmla v18.4s, v29.4s, v5.s[0]\n"
    "fmla v19.4s, v29.4s, v5.s[2]\n"
    "fmla v20.4s, v29.4s, v8.s[2]\n"
    "fmla v21.4s, v29.4s, v9.s[0]\n"
    "fmla v22.4s, v29.4s, v9.s[2]\n"
    "ldr q23, [%x[params], #0x20]\n"
    "fmla v14.4s, v25.4s, v2.s[0]\n"
    "fmla v15.4s, v25.4s, v2.s[2]\n"
    "fmla v16.4s, v25.4s, v3.s[0]\n"
    "fmla v17.4s, v25.4s, v6.s[0]\n"
    "fmla v18.4s, v25.4s, v6.s[2]\n"
    "fmla v19.4s, v25.4s, v7.s[0]\n"
    "fmla v20.4s, v25.4s, v10.s[0]\n"
    "fmla v21.4s, v25.4s, v10.s[2]\n"
    "fmla v22.4s, v25.4s, v11.s[0]\n"
    "ldr q25, [%x[params], #0x30]\n"
    "fmla v14.4s, v24.4s, v2.s[1]\n"
    "fmla v15.4s, v24.4s, v2.s[3]\n"
    "fmla v16.4s, v24.4s, v3.s[1]\n"
    "fmla v17.4s, v24.4s, v6.s[1]\n"
    "fmla v18.4s, v24.4s, v6.s[3]\n"
    "fmla v19.4s, v24.4s, v7.s[1]\n"
    "fmla v20.4s, v24.4s, v10.s[1]\n"
    "fmla v21.4s, v24.4s, v10.s[3]\n"
    "fmla v22.4s, v24.4s, v11.s[1]\n"
    "ldr q24, [%x[params], #0x40]\n"
    "fmla v14.4s, v23.4s, v2.s[2]\n"
    "fmla v15.4s, v23.4s, v3.s[0]\n"
    "fmla v16.4s, v23.4s, v3.s[2]\n"
    "fmla v17.4s, v23.4s, v6.s[2]\n"
    "fmla v18.4s, v23.4s, v7.s[0]\n"
    "fmla v19.4s, v23.4s, v7.s[2]\n"
    "fmla v20.4s, v23.4s, v10.s[2]\n"
    "fmla v21.4s, v23.4s, v11.s[0]\n"
    "fmla v22.4s, v23.4s, v11.s[2]\n"
    "ldr q23, [%x[params], #0x50]\n"
    "fmla v14.4s, v25.4s, v4.s[0]\n"
    "fmla v15.4s, v25.4s, v4.s[2]\n"
    "fmla v16.4s, v25.4s, v5.s[0]\n"
    "fmla v17.4s, v25.4s, v8.s[0]\n"
    "fmla v18.4s, v25.4s, v8.s[2]\n"
    "fmla v19.4s, v25.4s, v9.s[0]\n"
    "fmla v20.4s, v25.4s, v12.s[0]\n"
    "fmla v21.4s, v25.4s, v12.s[2]\n"
    "fmla v22.4s, v25.4s, v13.s[0]\n"
    "ldr q31, [%x[params], #0x70]\n"
    "fmla v14.4s, v24.4s, v4.s[1]\n"
    "fmla v15.4s, v24.4s, v4.s[3]\n"
    "fmla v16.4s, v24.4s, v5.s[1]\n"
    "fmla v17.4s, v24.4s, v8.s[1]\n"
    "fmla v18.4s, v24.4s, v8.s[3]\n"
    "fmla v19.4s, v24.4s, v9.s[1]\n"
    "fmla v20.4s, v24.4s, v12.s[1]\n"
    "fmla v21.4s, v24.4s, v12.s[3]\n"
    "fmla v22.4s, v24.4s, v13.s[1]\n"
    "ldr q30, [%x[params], #0x80]\n"
    "fmla v14.4s, v23.4s, v4.s[2]\n"
    "fmla v15.4s, v23.4s, v5.s[0]\n"
    "fmin v14.4s, v14.4s, v26.4s\n"
    "fmla v16.4s, v23.4s, v5.s[2]\n"
    "fmla v17.4s, v23.4s, v8.s[2]\n"
    "fmax v14.4s, v14.4s, v27.4s\n"
    "str q14, [x13, x14]\n"
    "ldr q14, [%x[params], #0x60]\n"
    "fmla v18.4s, v23.4s, v9.s[0]\n"
    "fmla v19.4s, v23.4s, v9.s[2]\n"
    "fmin v15.4s, v15.4s, v26.4s\n"
    "fmla v20.4s, v23.4s, v12.s[2]\n"
    "fmla v21.4s, v23.4s, v13.s[0]\n"
    "fmin v16.4s, v16.4s, v26.4s\n"
    "fmla v22.4s, v23.4s, v13.s[2]\n"
    "ldr q29, [%x[params], #0x90]\n"
    "fmin v17.4s, v17.4s, v26.4s\n"
    "add %x[params], %x[params], #0xa0\n"
    "fmin v18.4s, v18.4s, v26.4s\n"
    "fmin v19.4s, v19.4s, v26.4s\n"
    "fmin v20.4s, v20.4s, v26.4s\n"
    "fmin v21.4s, v21.4s, v26.4s\n"
    "fmin v22.4s, v22.4s, v26.4s\n"
    "fmax v15.4s, v15.4s, v27.4s\n"
    "str q15, [x12, x14]\n"
    "fmax v16.4s, v16.4s, v27.4s\n"
    "fmax v17.4s, v17.4s, v27.4s\n"
    "str q16, [x11, x14]\n"
    "fmax v18.4s, v18.4s, v27.4s\n"
    "fmax v19.4s, v19.4s, v27.4s\n"
    "str q17, [x10, x14]\n"
    "fmax v20.4s, v20.4s, v27.4s\n"
    "fmax v21.4s, v21.4s, v27.4s\n"
    "str q18, [x9, x14]\n"
    "fmax v22.4s, v22.4s, v27.4s\n"
    "str q19, [x28, x14]\n"
    "mov v15.16b, v14.16b\n"
    "str q20, [x27, x14]\n"
    "mov v16.16b, v14.16b\n"
    "mov v17.16b, v14.16b\n"
    "str q21, [x26, x14]\n"
    "mov v18.16b, v14.16b\n"
    "mov v19.16b, v14.16b\n"
    "str q22, [x25, x14]\n"
    "mov v20.16b, v14.16b\n"
    "mov v21.16b, v14.16b\n"
    "add x14, x14, #0x10\n"
    "mov v22.16b, v14.16b\n"
    "bgt 1b\n"
    "2:"  // Output channel complete vector tail
    "fmla v14.4s, v31.4s, v0.s[0]\n"
    "fmla v15.4s, v31.4s, v0.s[2]\n"
    "fmla v16.4s, v31.4s, v1.s[0]\n"
    "fmla v17.4s, v31.4s, v4.s[0]\n"
    "fmla v18.4s, v31.4s, v4.s[2]\n"
    "fmla v19.4s, v31.4s, v5.s[0]\n"
    "fmla v20.4s, v31.4s, v8.s[0]\n"
    "fmla v21.4s, v31.4s, v8.s[2]\n"
    "fmla v22.4s, v31.4s, v9.s[0]\n"
    "ldr q25, [%x[params], #0x0]\n"
    "fmla v14.4s, v30.4s, v0.s[1]\n"
    "fmla v15.4s, v30.4s, v0.s[3]\n"
    "fmla v16.4s, v30.4s, v1.s[1]\n"
    "fmla v17.4s, v30.4s, v4.s[1]\n"
    "fmla v18.4s, v30.4s, v4.s[3]\n"
    "fmla v19.4s, v30.4s, v5.s[1]\n"
    "fmla v20.4s, v30.4s, v8.s[1]\n"
    "fmla v21.4s, v30.4s, v8.s[3]\n"
    "fmla v22.4s, v30.4s, v9.s[1]\n"
    "ldr q24, [%x[params], #0x10]\n"
    "fmla v14.4s, v29.4s, v0.s[2]\n"
    "fmla v15.4s, v29.4s, v1.s[0]\n"
    "fmla v16.4s, v29.4s, v1.s[2]\n"
    "fmla v17.4s, v29.4s, v4.s[2]\n"
    "fmla v18.4s, v29.4s, v5.s[0]\n"
    "fmla v19.4s, v29.4s, v5.s[2]\n"
    "fmla v20.4s, v29.4s, v8.s[2]\n"
    "fmla v21.4s, v29.4s, v9.s[0]\n"
    "fmla v22.4s, v29.4s, v9.s[2]\n"
    "ldr q23, [%x[params], #0x20]\n"
    "fmla v14.4s, v25.4s, v2.s[0]\n"
    "fmla v15.4s, v25.4s, v2.s[2]\n"
    "fmla v16.4s, v25.4s, v3.s[0]\n"
    "fmla v17.4s, v25.4s, v6.s[0]\n"
    "fmla v18.4s, v25.4s, v6.s[2]\n"
    "fmla v19.4s, v25.4s, v7.s[0]\n"
    "fmla v20.4s, v25.4s, v10.s[0]\n"
    "fmla v21.4s, v25.4s, v10.s[2]\n"
    "fmla v22.4s, v25.4s, v11.s[0]\n"
    "ldr q25, [%x[params], #0x30]\n"
    "fmla v14.4s, v24.4s, v2.s[1]\n"
    "fmla v15.4s, v24.4s, v2.s[3]\n"
    "fmla v16.4s, v24.4s, v3.s[1]\n"
    "fmla v17.4s, v24.4s, v6.s[1]\n"
    "fmla v18.4s, v24.4s, v6.s[3]\n"
    "fmla v19.4s, v24.4s, v7.s[1]\n"
    "fmla v20.4s, v24.4s, v10.s[1]\n"
    "fmla v21.4s, v24.4s, v10.s[3]\n"
    "fmla v22.4s, v24.4s, v11.s[1]\n"
    "ldr q24, [%x[params], #0x40]\n"
    "fmla v14.4s, v23.4s, v2.s[2]\n"
    "fmla v15.4s, v23.4s, v3.s[0]\n"
    "fmla v16.4s, v23.4s, v3.s[2]\n"
    "fmla v17.4s, v23.4s, v6.s[2]\n"
    "fmla v18.4s, v23.4s, v7.s[0]\n"
    "fmla v19.4s, v23.4s, v7.s[2]\n"
    "fmla v20.4s, v23.4s, v10.s[2]\n"
    "fmla v21.4s, v23.4s, v11.s[0]\n"
    "fmla v22.4s, v23.4s, v11.s[2]\n"
    "ldr q23, [%x[params], #0x50]\n"
    "add %x[params], %x[params], #0x60\n"
    "fmla v14.4s, v25.4s, v4.s[0]\n"
    "fmla v15.4s, v25.4s, v4.s[2]\n"
    "fmla v16.4s, v25.4s, v5.s[0]\n"
    "fmla v17.4s, v25.4s, v8.s[0]\n"
    "fmla v18.4s, v25.4s, v8.s[2]\n"
    "fmla v19.4s, v25.4s, v9.s[0]\n"
    "fmla v20.4s, v25.4s, v12.s[0]\n"
    "fmla v21.4s, v25.4s, v12.s[2]\n"
    "fmla v22.4s, v25.4s, v13.s[0]\n"
    "fmla v14.4s, v24.4s, v4.s[1]\n"
    "fmla v15.4s, v24.4s, v4.s[3]\n"
    "fmla v16.4s, v24.4s, v5.s[1]\n"
    "fmla v17.4s, v24.4s, v8.s[1]\n"
    "fmla v18.4s, v24.4s, v8.s[3]\n"
    "fmla v19.4s, v24.4s, v9.s[1]\n"
    "fmla v20.4s, v24.4s, v12.s[1]\n"
    "fmla v21.4s, v24.4s, v12.s[3]\n"
    "fmla v22.4s, v24.4s, v13.s[1]\n"
    "fmla v14.4s, v23.4s, v4.s[2]\n"
    "fmla v15.4s, v23.4s, v5.s[0]\n"
    "fmin v14.4s, v14.4s, v26.4s\n"
    "fmla v16.4s, v23.4s, v5.s[2]\n"
    "fmla v17.4s, v23.4s, v8.s[2]\n"
    "fmin v15.4s, v15.4s, v26.4s\n"
    "fmla v18.4s, v23.4s, v9.s[0]\n"
    "fmla v19.4s, v23.4s, v9.s[2]\n"
    "fmin v16.4s, v16.4s, v26.4s\n"
    "fmla v20.4s, v23.4s, v12.s[2]\n"
    "fmla v21.4s, v23.4s, v13.s[0]\n"
    "fmin v17.4s, v17.4s, v26.4s\n"
    "fmla v22.4s, v23.4s, v13.s[2]\n"
    "fmin v18.4s, v18.4s, v26.4s\n"
    "fmin v19.4s, v19.4s, v26.4s\n"
    "fmin v20.4s, v20.4s, v26.4s\n"
    "fmin v21.4s, v21.4s, v26.4s\n"
    "fmin v22.4s, v22.4s, v26.4s\n"
    "fmax v14.4s, v14.4s, v27.4s\n"
    "fmax v15.4s, v15.4s, v27.4s\n"
    "str q14, [x13, x14]\n"
    "fmax v16.4s, v16.4s, v27.4s\n"
    "fmax v17.4s, v17.4s, v27.4s\n"
    "str q15, [x12, x14]\n"
    "fmax v18.4s, v18.4s, v27.4s\n"
    "fmax v19.4s, v19.4s, v27.4s\n"
    "str q16, [x11, x14]\n"
    "fmax v20.4s, v20.4s, v27.4s\n"
    "fmax v21.4s, v21.4s, v27.4s\n"
    "str q17, [x10, x14]\n"
    "fmax v22.4s, v22.4s, v27.4s\n"
    "str q18, [x9, x14]\n"
    "str q19, [x28, x14]\n"
    "str q20, [x27, x14]\n"
    "str q21, [x26, x14]\n"
    "str q22, [x25, x14]\n"
    "add x14, x14, #0x10\n"
    "3:"  // Output channel oddments
    "tst %x[channel_multiplier], #0x3\n"
    "beq 6f\n"
    "ldr q14, [%x[params], #0x0]\n"
    "ldr q25, [%x[params], #0x10]\n"
    "mov v15.16b, v14.16b\n"
    "mov v16.16b, v14.16b\n"
    "ldr q24, [%x[params], #0x20]\n"
    "ldr q23, [%x[params], #0x30]\n"
    "mov v17.16b, v14.16b\n"
    "mov v18.16b, v14.16b\n"
    "mov v19.16b, v14.16b\n"
    "mov v20.16b, v14.16b\n"
    "fmla v15.4s, v25.4s, v0.s[2]\n"
    "mov v21.16b, v14.16b\n"
    "mov v22.16b, v14.16b\n"
    "fmla v14.4s, v25.4s, v0.s[0]\n"
    "fmla v16.4s, v25.4s, v1.s[0]\n"
    "fmla v17.4s, v25.4s, v4.s[0]\n"
    "fmla v18.4s, v25.4s, v4.s[2]\n"
    "fmla v19.4s, v25.4s, v5.s[0]\n"
    "fmla v20.4s, v25.4s, v8.s[0]\n"
    "fmla v21.4s, v25.4s, v8.s[2]\n"
    "fmla v22.4s, v25.4s, v9.s[0]\n"
    "ldr q25, [%x[params], #0x40]\n"
    "fmla v14.4s, v24.4s, v0.s[1]\n"
    "fmla v15.4s, v24.4s, v0.s[3]\n"
    "fmla v16.4s, v24.4s, v1.s[1]\n"
    "fmla v17.4s, v24.4s, v4.s[1]\n"
    "fmla v18.4s, v24.4s, v4.s[3]\n"
    "fmla v19.4s, v24.4s, v5.s[1]\n"
    "fmla v20.4s, v24.4s, v8.s[1]\n"
    "fmla v21.4s, v24.4s, v8.s[3]\n"
    "fmla v22.4s, v24.4s, v9.s[1]\n"
    "ldr q24, [%x[params], #0x50]\n"
    "fmla v14.4s, v23.4s, v0.s[2]\n"
    "fmla v15.4s, v23.4s, v1.s[0]\n"
    "fmla v16.4s, v23.4s, v1.s[2]\n"
    "fmla v17.4s, v23.4s, v4.s[2]\n"
    "fmla v18.4s, v23.4s, v5.s[0]\n"
    "fmla v19.4s, v23.4s, v5.s[2]\n"
    "fmla v20.4s, v23.4s, v8.s[2]\n"
    "fmla v21.4s, v23.4s, v9.s[0]\n"
    "fmla v22.4s, v23.4s, v9.s[2]\n"
    "ldr q23, [%x[params], #0x60]\n"
    "fmla v14.4s, v25.4s, v2.s[0]\n"
    "fmla v15.4s, v25.4s, v2.s[2]\n"
    "fmla v16.4s, v25.4s, v3.s[0]\n"
    "fmla v17.4s, v25.4s, v6.s[0]\n"
    "fmla v18.4s, v25.4s, v6.s[2]\n"
    "fmla v19.4s, v25.4s, v7.s[0]\n"
    "fmla v20.4s, v25.4s, v10.s[0]\n"
    "fmla v21.4s, v25.4s, v10.s[2]\n"
    "fmla v22.4s, v25.4s, v11.s[0]\n"
    "ldr q25, [%x[params], #0x70]\n"
    "fmla v14.4s, v24.4s, v2.s[1]\n"
    "fmla v15.4s, v24.4s, v2.s[3]\n"
    "fmla v16.4s, v24.4s, v3.s[1]\n"
    "fmla v17.4s, v24.4s, v6.s[1]\n"
    "fmla v18.4s, v24.4s, v6.s[3]\n"
    "fmla v19.4s, v24.4s, v7.s[1]\n"
    "fmla v20.4s, v24.4s, v10.s[1]\n"
    "fmla v21.4s, v24.4s, v10.s[3]\n"
    "fmla v22.4s, v24.4s, v11.s[1]\n"
    "ldr q24, [%x[params], #0x80]\n"
    "fmla v14.4s, v23.4s, v2.s[2]\n"
    "fmla v15.4s, v23.4s, v3.s[0]\n"
    "fmla v16.4s, v23.4s, v3.s[2]\n"
    "fmla v17.4s, v23.4s, v6.s[2]\n"
    "fmla v18.4s, v23.4s, v7.s[0]\n"
    "fmla v19.4s, v23.4s, v7.s[2]\n"
    "fmla v20.4s, v23.4s, v10.s[2]\n"
    "fmla v21.4s, v23.4s, v11.s[0]\n"
    "fmla v22.4s, v23.4s, v11.s[2]\n"
    "ldr q23, [%x[params], #0x90]\n"
    "add %x[params], %x[params], #0xa0\n"
    "fmla v14.4s, v25.4s, v4.s[0]\n"
    "fmla v15.4s, v25.4s, v4.s[2]\n"
    "fmla v16.4s, v25.4s, v5.s[0]\n"
    "fmla v17.4s, v25.4s, v8.s[0]\n"
    "fmla v18.4s, v25.4s, v8.s[2]\n"
    "fmla v19.4s, v25.4s, v9.s[0]\n"
    "fmla v20.4s, v25.4s, v12.s[0]\n"
    "fmla v21.4s, v25.4s, v12.s[2]\n"
    "fmla v22.4s, v25.4s, v13.s[0]\n"
    "fmla v14.4s, v24.4s, v4.s[1]\n"
    "fmla v15.4s, v24.4s, v4.s[3]\n"
    "fmla v16.4s, v24.4s, v5.s[1]\n"
    "fmla v17.4s, v24.4s, v8.s[1]\n"
    "fmla v18.4s, v24.4s, v8.s[3]\n"
    "fmla v19.4s, v24.4s, v9.s[1]\n"
    "fmla v20.4s, v24.4s, v12.s[1]\n"
    "fmla v21.4s, v24.4s, v12.s[3]\n"
    "fmla v22.4s, v24.4s, v13.s[1]\n"
    "fmla v14.4s, v23.4s, v4.s[2]\n"
    "fmla v15.4s, v23.4s, v5.s[0]\n"
    "fmin v14.4s, v14.4s, v26.4s\n"
    "fmla v16.4s, v23.4s, v5.s[2]\n"
    "fmla v17.4s, v23.4s, v8.s[2]\n"
    "fmin v15.4s, v15.4s, v26.4s\n"
    "fmla v18.4s, v23.4s, v9.s[0]\n"
    "fmla v19.4s, v23.4s, v9.s[2]\n"
    "fmin v16.4s, v16.4s, v26.4s\n"
    "fmla v20.4s, v23.4s, v12.s[2]\n"
    "fmla v21.4s, v23.4s, v13.s[0]\n"
    "fmin v17.4s, v17.4s, v26.4s\n"
    "fmla v22.4s, v23.4s, v13.s[2]\n"
    "fmin v18.4s, v18.4s, v26.4s\n"
    "fmin v19.4s, v19.4s, v26.4s\n"
    "fmin v20.4s, v20.4s, v26.4s\n"
    "fmin v21.4s, v21.4s, v26.4s\n"
    "fmin v22.4s, v22.4s, v26.4s\n"
    "fmax v14.4s, v14.4s, v27.4s\n"
    "fmax v15.4s, v15.4s, v27.4s\n"
    "fmax v16.4s, v16.4s, v27.4s\n"
    "fmax v17.4s, v17.4s, v27.4s\n"
    "fmax v18.4s, v18.4s, v27.4s\n"
    "fmax v19.4s, v19.4s, v27.4s\n"
    "fmax v20.4s, v20.4s, v27.4s\n"
    "fmax v21.4s, v21.4s, v27.4s\n"
    "fmax v22.4s, v22.4s, v27.4s\n"
    "tbz %x[channel_multiplier], #1, 4f\n"
    "add x20, x13, x14\n"
    "add x22, x12, x14\n"
    "st1 { v14.d }[0], [x20]\n"
    "add x21, x11, x14\n"
    "add x20, x10, x14\n"
    "st1 { v15.d }[0], [x22]\n"
    "add x24, x9, x14\n"
    "add x23, x28, x14\n"
    "st1 { v16.d }[0], [x21]\n"
    "add x22, x27, x14\n"
    "add x21, x26, x14\n"
    "st1 { v17.d }[0], [x20]\n"
    "add x20, x25, x14\n"
    "st1 { v18.d }[0], [x24]\n"
    "add x14, x14, #0x8\n"
    "st1 { v19.d }[0], [x23]\n"
    "st1 { v20.d }[0], [x22]\n"
    "st1 { v21.d }[0], [x21]\n"
    "st1 { v22.d }[0], [x20]\n"
    "tbz %x[channel_multiplier], #0, 5f\n"
    "add x20, x13, x14\n"
    "add x22, x12, x14\n"
    "st1 { v14.s }[2], [x20]\n"
    "add x21, x11, x14\n"
    "add x20, x10, x14\n"
    "st1 { v15.s }[2], [x22]\n"
    "add x24, x9, x14\n"
    "add x23, x28, x14\n"
    "st1 { v16.s }[2], [x21]\n"
    "add x22, x27, x14\n"
    "add x21, x26, x14\n"
    "st1 { v17.s }[2], [x20]\n"
    "add x20, x25, x14\n"
    "st1 { v18.s }[2], [x24]\n"
    "st1 { v19.s }[2], [x23]\n"
    "st1 { v20.s }[2], [x22]\n"
    "st1 { v21.s }[2], [x21]\n"
    "st1 { v22.s }[2], [x20]\n"
    "b 5f\n"
    "4:"  // Output channel oddments: Store: Bit 1: Unset
    "add x20, x13, x14\n"
    "add x22, x12, x14\n"
    "st1 { v14.s }[0], [x20]\n"
    "add x21, x11, x14\n"
    "add x20, x10, x14\n"
    "st1 { v15.s }[0], [x22]\n"
    "add x24, x9, x14\n"
    "add x23, x28, x14\n"
    "st1 { v16.s }[0], [x21]\n"
    "add x22, x27, x14\n"
    "add x21, x26, x14\n"
    "st1 { v17.s }[0], [x20]\n"
    "add x20, x25, x14\n"
    "st1 { v18.s }[0], [x24]\n"
    "st1 { v19.s }[0], [x23]\n"
    "st1 { v20.s }[0], [x22]\n"
    "st1 { v21.s }[0], [x21]\n"
    "st1 { v22.s }[0], [x20]\n"
    "5:"  // Output channel oddments: Store: Bit 1: End
    "6:"  // End
    : [params] "+&r" (params)
    : [channel_multiplier] "r" (n_output_channels), [clamps] "r" (minmax_vals), [inptrs] "r" (inptrs), [outptrs] "r" (outptrs)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v29", "v30", "v31", "x9", "x10", "x11", "x12", "x13", "x14", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__)