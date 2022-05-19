#include <iostream>
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEGelu.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GeluFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{

} // namespace
TEST_SUITE(NEON)
TEST_SUITE(Gelu)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10, 10), 1, DataType::F32),
                                                TensorInfo(TensorShape(20, 20), 1, DataType::F32),
                                                TensorInfo(TensorShape(10, 10), 1, DataType::F32)}),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(10, 20), 1, DataType::F32),
                                                TensorInfo(TensorShape(20, 20), 1, DataType::F32),
                                                TensorInfo(TensorShape(20, 20), 1, DataType::F32)})),
        framework::dataset::make("Expected", { true, true, false })),
        input_info, output_info, expected)
{
    const Status status = NEGelu::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEGeluFixture = GeluValidationFixture<Tensor, Accessor, NEGelu, T>;

FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGeluFixture<uint8_t>, 
                       framework::DatasetMode::ALL, 
                       combine(datasets::MediumShapes(), framework::dataset::make("DataType", DataType::U8)))
{
    validate(Accessor(_target), _reference);
}
//Large1x3Shapes MediumShapes - 3
//Large2DShapes Medium3DShapes - 2
TEST_SUITE_END() // Gelu
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
