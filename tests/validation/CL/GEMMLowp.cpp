/*
 * Copyright (c) 2017-2024 Arm Limited.
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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/GEMMLowpFusedOffsetOutputDataset.h"
#include "tests/datasets/LargeGEMMLowpDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SmallGEMMLowpDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GEMMLowpFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{

using framework::dataset::make;

namespace
{
constexpr AbsoluteTolerance<float> tolerance_quant(1); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
}
TEST_SUITE(CL)
TEST_SUITE(GEMMLowp)

TEST_SUITE(MatrixMultiplyCore)
using CLGEMMLowpMatrixMultiplyCoreFixture = GEMMLowpMatrixMultiplyCoreValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore>;
using CLGEMMLowpBatchedMatMulFixture      = GEMMLowpMatrixMultiplyCoreValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore, false, false, true>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyCoreFixture, framework::DatasetMode::ALL, datasets::SmallGEMMLowpDataset())
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpMatrixMultiplyCoreFixture, framework::DatasetMode::NIGHTLY, datasets::LargeGEMMLowpDataset())
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

using CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixtureBatchedUnsigned =
    GEMMLowpBatchedMatrixMultiplyCoreFusedOffsetOutputFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore, false, false, uint8_t, uint8_t, true>;
TEST_SUITE(BatchedMatMul)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixtureBatchedUnsigned, framework::DatasetMode::ALL,
    combine(datasets::SmallGEMMLowpFusedBatchedMatMulDataset(),
        make("DataType", { DataType::QASYMM8 }),
        make("reshape_b_only_on_first_run", { false })))
{
    validate(CLAccessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // QASYMM8

using CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixtureBatchedSigned =
    GEMMLowpBatchedMatrixMultiplyCoreFusedOffsetOutputFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore, false, false, int8_t, int8_t, true>;
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixtureBatchedSigned, framework::DatasetMode::ALL,
    combine(datasets::SmallGEMMLowpFusedBatchedMatMulDataset(),
        make("DataType", { DataType::QASYMM8_SIGNED }),
        make("reshape_b_only_on_first_run", { false })))
{
    validate(CLAccessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // BatchedMatMul

TEST_SUITE(FusedOffsetOutput)
TEST_SUITE(QASYMM8)
using CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputUint8Fixture = GEMMLowpMatrixMultiplyCoreFusedOffsetOutputValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputUint8Fixture, framework::DatasetMode::ALL,
    combine(datasets::SmallGEMMLowpFusedOffsetOutputUint8Dataset(),
        make("DataType", { DataType::QASYMM8 }),
        make("reshape_b_only_on_first_run", { true, false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_quant);
}

TEST_SUITE(Output3D)
using CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputOutput3DUint8Fixture =
    GEMMLowpMatrixMultiplyCoreFusedOffsetOutputValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore, false, true>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputOutput3DUint8Fixture, framework::DatasetMode::ALL,
    combine(datasets::SmallGEMMLowpFusedOffsetOutputOutput3DUint8Dataset(),
        make("DataType", { DataType::QASYMM8 }),
        make("reshape_b_only_on_first_run", { true, false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // Output3D

TEST_SUITE(InputOutput3D)
using CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputInputOutput3DUint8Fixture =
    GEMMLowpMatrixMultiplyCoreFusedOffsetOutputValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore, true, true>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputInputOutput3DUint8Fixture, framework::DatasetMode::ALL,
    combine(datasets::SmallGEMMLowpFusedOffsetOutputInputOutput3DUint8Dataset(),
        make("DataType", { DataType::QASYMM8 }),
        make("reshape_b_only_on_first_run", { true, false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // InputOutput3D

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputUint8Fixture, framework::DatasetMode::NIGHTLY,
    combine(datasets::LargeGEMMLowpFusedOffsetOutputUint8Dataset(),
        make("DataType", { DataType::QASYMM8 }),
        make("reshape_b_only_on_first_run", { true, false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
using CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputInt8Fixture =
    GEMMLowpMatrixMultiplyCoreFusedOffsetOutputValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore, false, false, int8_t, int8_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyCoreFusedOffsetOutputInt8Fixture, framework::DatasetMode::ALL,
    combine(datasets::SmallGEMMLowpFusedOffsetOutputInt8Dataset(),
        make("DataType", { DataType::QASYMM8_SIGNED }),
        make("reshape_b_only_on_first_run", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // FusedOffsetOutput

TEST_SUITE(Output3D)
using CLGEMMLowpMatrixMultiplyCoreOutput3DFixture = GEMMLowpMatrixMultiplyCoreValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore, false, true>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyCoreOutput3DFixture, framework::DatasetMode::PRECOMMIT, datasets::SmallGEMMLowpOutput3DDataset())
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpMatrixMultiplyCoreOutput3DFixture, framework::DatasetMode::NIGHTLY, datasets::LargeGEMMLowpOutput3DDataset())
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // Output3D

TEST_SUITE(InputOutput3D)
using CLGEMMLowpMatrixMultiplyCoreInputOutput3DFixture = GEMMLowpMatrixMultiplyCoreValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore, true, true>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyCoreInputOutput3DFixture, framework::DatasetMode::PRECOMMIT, datasets::SmallGEMMLowpInputOutput3DDataset())
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpMatrixMultiplyCoreInputOutput3DFixture, framework::DatasetMode::NIGHTLY, datasets::LargeGEMMLowpInputOutput3DDataset())
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // InputOutput3D
TEST_SUITE_END() // MatrixMultiplyCore

TEST_SUITE(OutputStage)

TEST_SUITE(QuantizeDownInt32Scale)

TEST_SUITE(QASYMM8)

const auto quantize_down_int32_to_uint8_scale_cases = make("result_offset", -2, 1) * make("result_mult_int", 1, 2) * make("result_shift", 2, 3)
                                                      * make("min", 0) * make("max", 255) * make("addBias", { false, true });

const auto quantize_down_int32_to_uint8_scale_relu_cases = make("result_offset", -2, 1) * make("result_mult_int", 1, 2)
                                                           * make("result_shift", 2, 3) * make("min", 0, 2) * make("max", 171, 173) * make("addBias", { false, true });

using CLGEMMLowpQuantizeDownInt32ScaleFixture = GEMMLowpQuantizeDownInt32ToUint8ScaleValidationFixture<CLTensor, CLAccessor, CLGEMMLowpOutputStage>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpQuantizeDownInt32ScaleFixture, framework::DatasetMode::ALL,
    combine(datasets::SmallShapes(), quantize_down_int32_to_uint8_scale_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE(BoundedReLu)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpQuantizeDownInt32ScaleFixture, framework::DatasetMode::ALL,
    combine(datasets::SmallShapes(), quantize_down_int32_to_uint8_scale_relu_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END() // BoundedReLu
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)

const auto quantize_down_int32_to_int8_scale_cases = make("result_offset", -2, 1) * make("result_mult_int", 1, 2) * make("result_shift", 2, 3)
                                                     * make("min", -128) * make("max", 127) * make("addBias", { false, true });

const auto quantize_down_int32_to_int8_scale_relu_cases = make("result_offset", -2, 1) * make("result_mult_int", 1, 2)
                                                          * make("result_shift", 2, 3) * make("min", -100, -98) * make("max", 71, 73) * make("addBias", { false, true });

using CLGEMMLowpQuantizeDownInt32ScaleFixture = GEMMLowpQuantizeDownInt32ToInt8ScaleValidationFixture<CLTensor, CLAccessor, CLGEMMLowpOutputStage>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpQuantizeDownInt32ScaleFixture, framework::DatasetMode::ALL,
    combine(datasets::SmallShapes(), quantize_down_int32_to_int8_scale_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE(BoundedReLu)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpQuantizeDownInt32ScaleFixture, framework::DatasetMode::ALL,
    combine(datasets::SmallShapes(), quantize_down_int32_to_int8_scale_relu_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END() // BoundedReLu
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // QuantizeDownInt32Scale

TEST_SUITE(QuantizeDownInt32ScaleByFloat)

TEST_SUITE(QASYMM8)
using CLGEMMLowpQuantizeDownInt32ScaleByFloatFixture =
    GEMMLowpQuantizeDownInt32ScaleByFloatValidationFixture<CLTensor, CLAccessor, CLGEMMLowpOutputStage, uint8_t>;

FIXTURE_DATA_TEST_CASE(RunTiny, CLGEMMLowpQuantizeDownInt32ScaleByFloatFixture, framework::DatasetMode::ALL,
    combine(
        make("DataType", DataType::QASYMM8),
        datasets::TinyShapes(),
        make("result_real_multiplier", 0.33f),
        make("result_offset", 2, 3),
        make("min", 0),
        make("max", 255),
        make("addBias", { false, true })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
using CLGEMMLowpQuantizeDownInt32ScaleByFloatFixture_Signed =
    GEMMLowpQuantizeDownInt32ScaleByFloatValidationFixture<CLTensor, CLAccessor, CLGEMMLowpOutputStage, int8_t>;
FIXTURE_DATA_TEST_CASE(RunTiny, CLGEMMLowpQuantizeDownInt32ScaleByFloatFixture_Signed, framework::DatasetMode::ALL,
    combine(
        make("DataType", DataType::QASYMM8_SIGNED),
        datasets::TinyShapes(),
        make("result_real_multiplier", 0.33f),
        make("result_offset", 2, 3),
        make("min", -128),
        make("max", 127),
        make("addBias", { false, true })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE_END() // QuantizeDownInt32ScaleByFloat

TEST_SUITE_END() // OutputStage
TEST_SUITE_END() // GEMMLowp
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
