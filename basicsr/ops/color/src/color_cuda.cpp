#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

torch::Tensor RGB2LAB_sRGB_ForwardLaucher(
    torch::Tensor image 
    );

torch::Tensor RGB2LAB_sRGB_BackwardLaucher(
    torch::Tensor output_grad, 
    torch::Tensor image 
    );

torch::Tensor LAB2RGB_sRGB_ForwardLaucher(
    torch::Tensor image 
    );

torch::Tensor LAB2RGB_sRGB_BackwardLaucher(
    torch::Tensor output_grad, 
    torch::Tensor image 
    );

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor rgb2lab_srgb_forward(
    torch::Tensor image
    ) {

    CHECK_INPUT(image);
    const at::cuda::OptionalCUDAGuard device_guard(image.device());

    return RGB2LAB_sRGB_ForwardLaucher(image);
}

torch::Tensor rgb2lab_srgb_backward(
    torch::Tensor output_grad,
    torch::Tensor image
    ) {

    CHECK_INPUT(output_grad);
    CHECK_INPUT(image);
    const at::cuda::OptionalCUDAGuard device_guard(image.device());

    return RGB2LAB_sRGB_BackwardLaucher(output_grad,image);
}

torch::Tensor lab2rgb_srgb_forward(
    torch::Tensor image
    ) {

    CHECK_INPUT(image);
    const at::cuda::OptionalCUDAGuard device_guard(image.device());

    return LAB2RGB_sRGB_ForwardLaucher(image);
}

torch::Tensor lab2rgb_srgb_backward(
    torch::Tensor output_grad,
    torch::Tensor image
    ) {

    CHECK_INPUT(output_grad);
    CHECK_INPUT(image);
    const at::cuda::OptionalCUDAGuard device_guard(image.device());

    return LAB2RGB_sRGB_BackwardLaucher(output_grad,image);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rgb2lab_srgb_forward", &rgb2lab_srgb_forward, "rgb2lab_srgb_forward (CUDA)");
    m.def("rgb2lab_srgb_backward", &rgb2lab_srgb_backward, "rgb2lab_srgb_backward (CUDA)");
    m.def("lab2rgb_srgb_forward", &lab2rgb_srgb_forward, "lab2rgb_srgb_forward (CUDA)");
    m.def("lab2rgb_srgb_backward", &lab2rgb_srgb_backward, "lab2rgb_srgb_backward (CUDA)");
}
        
