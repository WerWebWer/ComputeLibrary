#ifndef ARM_COMPUTE_NEGELUKERNEL_H
#define ARM_COMPUTE_NEGELUKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Basic kernel to perform a tile operation */
class NEGeluKernel : public INEKernel
{
public:
    /** Default constructor */
    NEGeluKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    NEGeluKernel(const NEGeluKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    NEGeluKernel &operator=(const NEGeluKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGeluKernel(NEGeluKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGeluKernel &operator=(NEGeluKernel &&) = default;
    /** Default destructor */
    ~NEGeluKernel() = default;
    const char *name() const override
    {
        return "NEGeluKernel";
    }

    void configure(const ITensor *input, ITensor *output);

    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor         *_input;
    ITensor               *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGELUKERNEL_H */
