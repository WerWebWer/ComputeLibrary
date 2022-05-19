#include "arm_compute/runtime/NEON/functions/NEGelu.h"

#include "src/core/NEON/kernels/NEGeluKernel.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
void NEGelu::configure(const ITensor *input, ITensor *output) {
    ARM_COMPUTE_LOG_PARAMS(input, output);

    auto k = std::make_unique<NEGeluKernel>();
    k->configure(input, output);
    _kernel = std::move(k);
}

Status NEGelu::validate(const ITensorInfo *input, const ITensorInfo *output) {
    return NEGeluKernel::validate(input, output);
}
} // namespace arm_compute
