#ifndef ARM_COMPUTE_TEST_GELU_H
#define ARM_COMPUTE_TEST_GELU_H

#include "tests/SimpleTensor.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> gelu(const SimpleTensor<T> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GELU_H */
