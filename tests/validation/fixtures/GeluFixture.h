#ifndef ARM_COMPUTE_TEST_GELU_FIXTURE
#define ARM_COMPUTE_TEST_GELU_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Gelu.h"

#include <omp.h>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class GeluValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type) {
        _target    = compute_target(shape, data_type);
        _reference = compute_reference(shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor) {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type)
    {
        double start, end;
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type);
        TensorType dst = create_tensor<TensorType>(shape, data_type);

        // Create and configure function
        FunctionType gelu_func;
        gelu_func.configure(&src, &dst);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src));

#if defined(_OPENMP)
        start = omp_get_wtime();
#endif /* _OPENMP */
        gelu_func.run();
#if defined(_OPENMP)
        end = omp_get_wtime();
        std::cout << "Targe time     " << end - start << std::endl;
#endif /* _OPENMP */

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type)
    {
        double start, end;
        // Create reference
        SimpleTensor<T> src{ shape, data_type };
        SimpleTensor<T> dst;

        // Fill reference
        fill(src);

#if defined(_OPENMP)
        start = omp_get_wtime();
#endif /* _OPENMP */
        dst = reference::gelu<T>(src);
#if defined(_OPENMP)
        end = omp_get_wtime();
        std::cout << "Reference time " << end - start << std::endl;
#endif /* _OPENMP */

        return dst;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GELU_FIXTURE */
