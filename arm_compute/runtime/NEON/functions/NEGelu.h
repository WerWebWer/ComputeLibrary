#ifndef ARM_COMPUTE_NEGELU_H
#define ARM_COMPUTE_NEGELU_H

#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to run @ref NEGeluKernel */
class NEGelu : public INESimpleFunctionNoBorder
{
public:

    void configure(const ITensor *input, ITensor *output);

    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEGELU_H */
