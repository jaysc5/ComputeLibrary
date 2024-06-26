/*
 * Copyright (c) 2024 Arm Limited.
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

#pragma once

#if defined(__aarch64__)

namespace {

void a64_transpose_interleave_4_2x4_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 4 * roundup<size_t>(height, 4) * sizeof(bfloat16);

    __asm__ __volatile__(
      "cmp %x[height], #0x8\n"
      "blt 8f\n"
      "1:"  // Main row loop: Head
      "mov x9, %x[in]\n"
      "mov x28, %x[width]\n"
      "mov x27, %x[out]\n"
      "sub %x[height], %x[height], #0x8\n"
      "add x26, x9, %x[in_stride]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "cmp x28, #0x8\n"
      "add x23, x24, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add x20, x21, %x[in_stride]\n"
      "add %x[in], x20, %x[in_stride]\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ldr q19, [x9], #0x10\n"
      "ldr q18, [x26], #0x10\n"
      "sub x28, x28, #0x8\n"
      "ldr q17, [x25], #0x10\n"
      "ldr q16, [x24], #0x10\n"
      "cmp x28, #0x8\n"
      "ldr q1, [x23], #0x10\n"
      "ldr q0, [x22], #0x10\n"
      "ldr q31, [x21], #0x10\n"
      "ldr q24, [x20], #0x10\n"
      "ldr q23, [x9], #0x10\n"
      "ldr q22, [x26], #0x10\n"
      "zip1 v30.4s, v19.4s, v17.4s\n"
      "zip1 v29.4s, v18.4s, v16.4s\n"
      "ldr q21, [x25], #0x10\n"
      "ldr q20, [x24], #0x10\n"
      "zip2 v28.4s, v19.4s, v17.4s\n"
      "zip2 v27.4s, v18.4s, v16.4s\n"
      "ldr q19, [x23], #0x10\n"
      "ldr q18, [x22], #0x10\n"
      "zip1 v26.4s, v1.4s, v31.4s\n"
      "zip1 v25.4s, v0.4s, v24.4s\n"
      "ldr q17, [x21], #0x10\n"
      "ldr q16, [x20], #0x10\n"
      "zip2 v8.4s, v1.4s, v31.4s\n"
      "zip2 v24.4s, v0.4s, v24.4s\n"
      "zip1 v7.4s, v23.4s, v21.4s\n"
      "zip1 v6.4s, v22.4s, v20.4s\n"
      "zip2 v5.4s, v23.4s, v21.4s\n"
      "zip2 v4.4s, v22.4s, v20.4s\n"
      "zip1 v3.4s, v19.4s, v17.4s\n"
      "zip1 v2.4s, v18.4s, v16.4s\n"
      "zip2 v1.4s, v19.4s, v17.4s\n"
      "zip2 v0.4s, v18.4s, v16.4s\n"
      "zip1 v23.4s, v30.4s, v29.4s\n"
      "zip1 v22.4s, v28.4s, v27.4s\n"
      "zip1 v21.4s, v26.4s, v25.4s\n"
      "zip1 v20.4s, v8.4s, v24.4s\n"
      "zip1 v19.4s, v7.4s, v6.4s\n"
      "zip1 v18.4s, v5.4s, v4.4s\n"
      "zip1 v17.4s, v3.4s, v2.4s\n"
      "zip1 v16.4s, v1.4s, v0.4s\n"
      ".inst 0x0ea16aff  // bfcvtn v31.4h, v23.4s\n"
      "zip2 v30.4s, v30.4s, v29.4s\n"
      ".inst 0x0ea16add  // bfcvtn v29.4h, v22.4s\n"
      "zip2 v28.4s, v28.4s, v27.4s\n"
      ".inst 0x0ea16abb  // bfcvtn v27.4h, v21.4s\n"
      "zip2 v26.4s, v26.4s, v25.4s\n"
      ".inst 0x0ea16a99  // bfcvtn v25.4h, v20.4s\n"
      "zip2 v24.4s, v8.4s, v24.4s\n"
      ".inst 0x0ea16a77  // bfcvtn v23.4h, v19.4s\n"
      "zip2 v22.4s, v7.4s, v6.4s\n"
      ".inst 0x0ea16a55  // bfcvtn v21.4h, v18.4s\n"
      "zip2 v20.4s, v5.4s, v4.4s\n"
      ".inst 0x0ea16a33  // bfcvtn v19.4h, v17.4s\n"
      "zip2 v18.4s, v3.4s, v2.4s\n"
      ".inst 0x0ea16a11  // bfcvtn v17.4h, v16.4s\n"
      "zip2 v16.4s, v1.4s, v0.4s\n"
      ".inst 0x4ea16bdf  // bfcvtn2 v31.8h, v30.4s\n"
      ".inst 0x4ea16b9d  // bfcvtn2 v29.8h, v28.4s\n"
      ".inst 0x4ea16b5b  // bfcvtn2 v27.8h, v26.4s\n"
      ".inst 0x4ea16b19  // bfcvtn2 v25.8h, v24.4s\n"
      ".inst 0x4ea16ad7  // bfcvtn2 v23.8h, v22.4s\n"
      ".inst 0x4ea16a95  // bfcvtn2 v21.8h, v20.4s\n"
      "str q31, [x27, #0x0]\n"
      "str q29, [x27, #0x10]\n"
      ".inst 0x4ea16a53  // bfcvtn2 v19.8h, v18.4s\n"
      ".inst 0x4ea16a11  // bfcvtn2 v17.8h, v16.4s\n"
      "str q27, [x27, #0x20]\n"
      "str q25, [x27, #0x30]\n"
      "add x27, x27, %x[out_stride]\n"
      "str q23, [x27, #0x0]\n"
      "str q21, [x27, #0x10]\n"
      "str q19, [x27, #0x20]\n"
      "str q17, [x27, #0x30]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cmp x28, #0x4\n"
      "blt 5f\n"
      "4:"  // Main row loop: Column loop
      "ldr q25, [x9], #0x10\n"
      "ldr q24, [x26], #0x10\n"
      "sub x28, x28, #0x4\n"
      "ldr q21, [x25], #0x10\n"
      "ldr q20, [x24], #0x10\n"
      "cmp x28, #0x4\n"
      "ldr q23, [x23], #0x10\n"
      "ldr q19, [x22], #0x10\n"
      "ldr q18, [x21], #0x10\n"
      "ldr q17, [x20], #0x10\n"
      "zip1 v22.4s, v25.4s, v21.4s\n"
      "zip1 v16.4s, v24.4s, v20.4s\n"
      "zip2 v21.4s, v25.4s, v21.4s\n"
      "zip2 v20.4s, v24.4s, v20.4s\n"
      "zip1 v27.4s, v23.4s, v18.4s\n"
      "zip1 v26.4s, v19.4s, v17.4s\n"
      "zip2 v25.4s, v23.4s, v18.4s\n"
      "zip2 v24.4s, v19.4s, v17.4s\n"
      "zip1 v19.4s, v22.4s, v16.4s\n"
      "zip1 v18.4s, v21.4s, v20.4s\n"
      "zip1 v17.4s, v27.4s, v26.4s\n"
      "zip2 v23.4s, v22.4s, v16.4s\n"
      "zip1 v16.4s, v25.4s, v24.4s\n"
      "zip2 v22.4s, v21.4s, v20.4s\n"
      ".inst 0x0ea16a75  // bfcvtn v21.4h, v19.4s\n"
      ".inst 0x0ea16a54  // bfcvtn v20.4h, v18.4s\n"
      ".inst 0x0ea16a33  // bfcvtn v19.4h, v17.4s\n"
      "zip2 v18.4s, v27.4s, v26.4s\n"
      ".inst 0x0ea16a11  // bfcvtn v17.4h, v16.4s\n"
      "zip2 v16.4s, v25.4s, v24.4s\n"
      ".inst 0x4ea16af5  // bfcvtn2 v21.8h, v23.4s\n"
      ".inst 0x4ea16ad4  // bfcvtn2 v20.8h, v22.4s\n"
      ".inst 0x4ea16a53  // bfcvtn2 v19.8h, v18.4s\n"
      ".inst 0x4ea16a11  // bfcvtn2 v17.8h, v16.4s\n"
      "str q21, [x27, #0x0]\n"
      "str q20, [x27, #0x10]\n"
      "str q19, [x27, #0x20]\n"
      "str q17, [x27, #0x30]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cbz x28, 7f\n"
      "movi v16.16b, #0x0\n"
      "str q16, [x27, #0x0]\n"
      "str q16, [x27, #0x10]\n"
      "str q16, [x27, #0x20]\n"
      "str q16, [x27, #0x30]\n"
      "6:"  // Main row loop: width 1 loop: loop
      "ldr s23, [x9], #0x4\n"
      "ldr s22, [x26], #0x4\n"
      "sub x28, x28, #0x1\n"
      "ldr s19, [x25], #0x4\n"
      "ldr s17, [x24], #0x4\n"
      "cmp x28, #0x1\n"
      "ldr s21, [x23], #0x4\n"
      "ldr s20, [x22], #0x4\n"
      "ldr s18, [x21], #0x4\n"
      "ldr s16, [x20], #0x4\n"
      "zip1 v19.4s, v23.4s, v19.4s\n"
      "zip1 v17.4s, v22.4s, v17.4s\n"
      "zip1 v18.4s, v21.4s, v18.4s\n"
      "zip1 v16.4s, v20.4s, v16.4s\n"
      "zip1 v17.4s, v19.4s, v17.4s\n"
      "zip1 v16.4s, v18.4s, v16.4s\n"
      ".inst 0x0ea16a31  // bfcvtn v17.4h, v17.4s\n"
      ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
      "str d17, [x27, #0x0]\n"
      "str d16, [x27, #0x20]\n"
      "add x27, x27, #0x8\n"
      "bge 6b\n"
      "7:"  // Main row loop: odd col skip
      "cmp %x[height], #0x8\n"
      "add %x[out], %x[out], #0x40\n"
      "bge 1b\n"
      "cbz %x[height], 16f\n"
      "8:"  // Main loop skip
      "9:"  // Tail row loop: Head
      "mov x9, %x[in]\n"
      "mov x20, %x[width]\n"
      "cmp %x[height], #0x3\n"
      "mov x27, %x[out]\n"
      "add x26, x9, %x[in_stride]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "csel x25, x25, %x[pad_row], GE\n"
      "add %x[in], x24, %x[in_stride]\n"
      "csel x24, x24, %x[pad_row], GT\n"
      "cmp %x[height], #0x1\n"
      "sub %x[height], %x[height], #0x4\n"
      "csel x26, x26, %x[pad_row], GT\n"
      "cmp x20, #0x8\n"
      "blt 11f\n"
      "10:"  // Tail row loop: Unroll column loop
      "ldr q25, [x9], #0x10\n"
      "ldr q24, [x26], #0x10\n"
      "sub x20, x20, #0x8\n"
      "ldr q21, [x25], #0x10\n"
      "ldr q20, [x24], #0x10\n"
      "cmp x20, #0x8\n"
      "ldr q23, [x9], #0x10\n"
      "ldr q19, [x26], #0x10\n"
      "ldr q18, [x25], #0x10\n"
      "ldr q17, [x24], #0x10\n"
      "zip1 v22.4s, v25.4s, v21.4s\n"
      "zip1 v16.4s, v24.4s, v20.4s\n"
      "zip2 v21.4s, v25.4s, v21.4s\n"
      "zip2 v20.4s, v24.4s, v20.4s\n"
      "zip1 v27.4s, v23.4s, v18.4s\n"
      "zip1 v26.4s, v19.4s, v17.4s\n"
      "zip2 v25.4s, v23.4s, v18.4s\n"
      "zip2 v24.4s, v19.4s, v17.4s\n"
      "zip1 v19.4s, v22.4s, v16.4s\n"
      "zip1 v18.4s, v21.4s, v20.4s\n"
      "zip1 v17.4s, v27.4s, v26.4s\n"
      "zip2 v23.4s, v22.4s, v16.4s\n"
      "zip1 v16.4s, v25.4s, v24.4s\n"
      "zip2 v22.4s, v21.4s, v20.4s\n"
      ".inst 0x0ea16a75  // bfcvtn v21.4h, v19.4s\n"
      ".inst 0x0ea16a54  // bfcvtn v20.4h, v18.4s\n"
      ".inst 0x0ea16a33  // bfcvtn v19.4h, v17.4s\n"
      "zip2 v18.4s, v27.4s, v26.4s\n"
      ".inst 0x0ea16a11  // bfcvtn v17.4h, v16.4s\n"
      "zip2 v16.4s, v25.4s, v24.4s\n"
      ".inst 0x4ea16af5  // bfcvtn2 v21.8h, v23.4s\n"
      ".inst 0x4ea16ad4  // bfcvtn2 v20.8h, v22.4s\n"
      ".inst 0x4ea16a53  // bfcvtn2 v19.8h, v18.4s\n"
      ".inst 0x4ea16a11  // bfcvtn2 v17.8h, v16.4s\n"
      "str q21, [x27, #0x0]\n"
      "str q20, [x27, #0x10]\n"
      "add x27, x27, %x[out_stride]\n"
      "str q19, [x27, #0x0]\n"
      "str q17, [x27, #0x10]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 10b\n"
      "11:"  // Tail row loop: Unroll column loop skip
      "cmp x20, #0x4\n"
      "blt 13f\n"
      "12:"  // Tail row loop: Column loop
      "ldr q21, [x9], #0x10\n"
      "ldr q20, [x26], #0x10\n"
      "sub x20, x20, #0x4\n"
      "ldr q19, [x25], #0x10\n"
      "ldr q17, [x24], #0x10\n"
      "cmp x20, #0x4\n"
      "zip1 v18.4s, v21.4s, v19.4s\n"
      "zip1 v16.4s, v20.4s, v17.4s\n"
      "zip2 v21.4s, v21.4s, v19.4s\n"
      "zip2 v20.4s, v20.4s, v17.4s\n"
      "zip1 v17.4s, v18.4s, v16.4s\n"
      "zip2 v19.4s, v18.4s, v16.4s\n"
      "zip1 v16.4s, v21.4s, v20.4s\n"
      ".inst 0x0ea16a32  // bfcvtn v18.4h, v17.4s\n"
      "zip2 v17.4s, v21.4s, v20.4s\n"
      ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
      ".inst 0x4ea16a72  // bfcvtn2 v18.8h, v19.4s\n"
      ".inst 0x4ea16a30  // bfcvtn2 v16.8h, v17.4s\n"
      "str q18, [x27, #0x0]\n"
      "str q16, [x27, #0x10]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 12b\n"
      "13:"  // Tail row loop: Column loop skip
      "cbz x20, 15f\n"
      "movi v16.16b, #0x0\n"
      "str q16, [x27, #0x0]\n"
      "str q16, [x27, #0x10]\n"
      "14:"  // Tail row loop: width 1 loop: loop
      "ldr s19, [x9], #0x4\n"
      "ldr s18, [x26], #0x4\n"
      "sub x20, x20, #0x1\n"
      "ldr s17, [x25], #0x4\n"
      "ldr s16, [x24], #0x4\n"
      "cmp x20, #0x1\n"
      "zip1 v17.4s, v19.4s, v17.4s\n"
      "zip1 v16.4s, v18.4s, v16.4s\n"
      "zip1 v16.4s, v17.4s, v16.4s\n"
      ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
      "str d16, [x27, #0x0]\n"
      "add x27, x27, #0x8\n"
      "bge 14b\n"
      "15:"  // Tail row loop: odd col skip
      "cmp %x[height], #0x1\n"
      "add %x[out], %x[out], #0x20\n"
      "bge 9b\n"
      "16:"  // Done
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}

} // anonymous namespace
template<>
void Transform<4, 4, true, VLType::None>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_4_2x4_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}


#endif  // defined(__aarch64__)
