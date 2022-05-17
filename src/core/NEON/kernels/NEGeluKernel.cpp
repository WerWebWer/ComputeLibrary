#include "src/core/NEON/kernels/NEGeluKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>
#include <omp.h>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    return Status{};
}
inline void gelu_openmp_U8(const uint8_t *__restrict input, uint8_t *__restrict output)
{
#if defined(_OPENMP)
    omp_set_num_threads(4);
    #pragma omp parallel for
#endif /* _OPENMP */
    for (int i = 0; i < 4; i++) {
        *(output+i) =  0.5 * (*(input+i)) * (1 + erf(*(input+i) / std::sqrt(2.0)));
    }
}
inline void gelu_neon_U8(const uint8_t *__restrict input, uint8_t *__restrict output)
{
    const uint8x16_t val = vld1q_u8(input);

    const uint8x16_t max9 = vmovq_n_u8(9); // vector with only 9
    const uint8x16_t min0 = vmovq_n_u8(0); // vector with only 0

    uint8x16_t mask1 = vcgtq_u8(val, min0); // > 0
    uint8x16_t mask2 = vcltq_u8(val, max9); // < 9
    uint8x16_t mask = vandq_u8(mask1, mask2); // mask1 & mask2

    uint8x16_t res = vaddq_u8(val, mask);

    vst1q_u8(output, res);
}
} // namespace

NEGeluKernel::NEGeluKernel() : _input(nullptr), _output(nullptr)
{
    std::cout << "NEGeluKernel::NEGeluKernel()" << std::endl;
}

void NEGeluKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    
    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    set_format_if_unknown(*output->info(), Format::U8);
    set_format_if_unknown(*input->info(), Format::U8);

    // Validate
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    constexpr unsigned int num_elems_processed_per_iteration = 4;

    // Configure kernel window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration), output_access);

    INEKernel::configure(win);
}

Status NEGeluKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    return Status{};
}

void NEGeluKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const auto src_shape = _input->info()->tensor_shape();
   
    Iterator input_it(_input, window);
    Iterator output_it(_output, window);
    execute_window_loop(window, [&](const Coordinates &) {
            gelu_neon_U8(input_it.ptr(), output_it.ptr());
        }, input_it, output_it);
    

    // std::ostream& os = std::cout;
    // std::cout << std::endl << "Print" << std::endl;
    // std::cout << "input" << std::endl;
    // _input->print(os, IOFormatInfo());
    // std::cout << "output" << std::endl;
    // _output->print(os,IOFormatInfo());
    // std::cout << std::endl;
}
} // namespace arm_compute
