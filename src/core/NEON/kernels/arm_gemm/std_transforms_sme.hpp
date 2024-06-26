/*
 * Copyright (c) 2022-2024 Arm Limited.
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

#include "interleave_indirect.hpp"
#include "transform.hpp"

namespace arm_gemm {

/*
 * Define "standard" transforms for the blocked GEMMs for SVE.
 *
 * This assumes that A is interleaved 'height' ways, B is interleaved
 * 'width'xVL ways and transposed, and that the merge needs to work in
 * 'height' x 'width'xVL blocks.
 *
 * The optional 'block' parameter is for kernels using dot-product type
 * instructions like UDOT and SDOT.
 */
template<typename TOperand, typename TResult, unsigned int height_vectors, unsigned int width_vectors, unsigned int block=1, bool integrate_sums=false>
class StdTransformsSME
{
public:
    template<typename TIn>
    void PrepareA(TOperand *out, const TIn *in, const int stride, const int y0,
                  const int ymax, const int k0, const int kmax, int32_t row_sum_multiplier) {
        Interleave<height_vectors, block, VLType::SME>(out, in, stride, y0, ymax, k0, kmax, integrate_sums, row_sum_multiplier);
    }

    template<typename TIn>
    void PrepareA_indirect(TOperand *out, const TIn * const * const *ptr, size_t stringlen, size_t rounded_stringlen, const int y0,
                           const int ymax, const int k0, const int kmax, int32_t row_sum_multiplier) {
        IndirectInterleave<height_vectors, block, VLType::SME>(out, ptr, stringlen, rounded_stringlen, y0, ymax, k0, kmax, integrate_sums, row_sum_multiplier);
    }

    template<typename TIn>
    void PrepareA_convolution(TOperand *out, const TIn *ptr, size_t stride, const convolver<TIn> &conv, size_t rounded_stringlen,
                              const int y0, const int ymax, const int k0, const int kmax, int32_t row_sum_multiplier) {
        ConvolutionInterleave<height_vectors, block, VLType::SME>(out, ptr, stride, conv, rounded_stringlen, y0, ymax, k0, kmax, integrate_sums, row_sum_multiplier);
    }

    bool PrepareB_supports_transpose() const {
        return false;
    }

    template<typename TIn>
    void PrepareB(TOperand *out, const TIn *in, const int stride, const int x0,
                  const int xmax, const int k0, const int kmax, bool transposed) {
        assert (!transposed);
        Transform<width_vectors, block,  true, VLType::SME>(out, in, stride, x0, xmax, k0, kmax);
    }

    template<typename TOut>
    void Merge(TOut *, const TResult *, int, int, int, int, int, const TOut *, const Activation, bool) {
        // Separate merge not supported for SME.
    }
};

} // namespace arm_gemm
