#include "Gelu.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
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
SimpleTensor<T> gelu(const SimpleTensor<T> &src) {
    SimpleTensor<T> dst(src.shape(), src.data_type());

    for(int i = 0; i < src.num_elements(); ++i) {
        dst[i] = 0.5 * src[i] * (1 + erf(src[i] / std::sqrt(2.0)));
    }

    // std::ostream& os = std::cout;
    // std::cout << std::endl << "Print reference" << std::endl;
    // std::cout << "input" << std::endl;
    // for (int i = 0; i < src.num_elements(); ++i) {
    //     std::cout << (int)src[i] << " ";
    // }
    // std::cout << std::endl << "output" << std::endl;
    // for (int i = 0; i < dst.num_elements(); ++i) {
    //     std::cout << (int)dst[i] << " ";
    // }
    // std::cout << std::endl << std::endl;

    return dst;
}
template SimpleTensor<uint8_t> gelu(const SimpleTensor<uint8_t> &src);
// template SimpleTensor<int8_t> gelu(const SimpleTensor<int8_t> &src);
// template SimpleTensor<uint16_t> gelu(const SimpleTensor<uint16_t> &src);
// template SimpleTensor<int16_t> gelu(const SimpleTensor<int16_t> &src);
// template SimpleTensor<uint32_t> gelu(const SimpleTensor<uint32_t> &src);
// template SimpleTensor<int32_t> gelu(const SimpleTensor<int32_t> &src);
// template SimpleTensor<half> gelu(const SimpleTensor<half> &src);
// template SimpleTensor<float> gelu(const SimpleTensor<float> &src);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
