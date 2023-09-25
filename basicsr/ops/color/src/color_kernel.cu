#include <stdio.h>
#include <math.h>
#include <float.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
    template <typename scalar_t>
    __device__ scalar_t* atomicAdd(scalar_t* a, scalar_t b) { 
        *a += b; 
        return a;
    }
#endif

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)

    template <typename scalar_t>
    __global__ void RGB2LAB_sRGB_Forward(
        const int nthreads, 
        const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image, 
        torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output, 
        const int batch, 
        const int height, 
        const int width, 
        const int channel) {

        CUDA_1D_KERNEL_LOOP(index, nthreads) {
        
        int b = index / (height * width);
        index %= height * width;
        int h = index / width;
        index %= width;
        int w = index;

        float R = image[b][h][w][0];
        float G = image[b][h][w][1];
        float B = image[b][h][w][2];
        float L, A, Bl, x, y, z;

        R = ((R > 0.04045) ? pow((R + 0.055f) / 1.055f, 2.4f) : (R / 12.92)) * 100.0;
        G = ((G > 0.04045) ? pow((G + 0.055f) / 1.055f, 2.4f) : (G / 12.92)) * 100.0;
        B = ((B > 0.04045) ? pow((B + 0.055f) / 1.055f, 2.4f) : (B / 12.92)) * 100.0;
        
        x = R*0.4124564 + G*0.3575761 + B*0.1804375;
        y = R*0.2126729 + G*0.7151522 + B*0.0721750;
        z = R*0.0193339 + G*0.1191920 + B*0.9503041;

        x = x / 95.047;
        y = y / 100.00;
        z = z / 108.883;
        
        x = (x > 0.008856) ? cbrt(x) : (7.787 * x + 16.0 / 116.0);
        y = (y > 0.008856) ? cbrt(y) : (7.787 * y + 16.0 / 116.0);
        z = (z > 0.008856) ? cbrt(z) : (7.787 * z + 16.0 / 116.0);
        
        L = (116.0 * y) - 16;
        A = 500 * (x - y);
        Bl = 200 * (y - z);

        output[b][h][w][0] = L;
        output[b][h][w][1] = A;
        output[b][h][w][2] = Bl;

        }
    }

    template <typename scalar_t>
    __global__ void RGB2LAB_sRGB_Backward(
        const int nthreads, 
        const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output_grad, 
        const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image, 
        torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image_grad, 
        const int batch, 
        const int height, 
        const int width, 
        const int channel) {

        CUDA_1D_KERNEL_LOOP(index, nthreads) {
        
        int b = index / (height * width);
        index %= height * width;
        int h = index / width;
        index %= width;
        int w = index;

        float R = image[b][h][w][0];
        float G = image[b][h][w][1];
        float B = image[b][h][w][2];
        float x, y, z;

        float gR, gG, gB, gx, gy, gz;

        gR = ((R > 0.04045) ? pow((R + 0.055f) / 1.055f, 1.4f)*2.4f*1/1.055f : 1/12.92) * 100.0;
        gG = ((G > 0.04045) ? pow((G + 0.055f) / 1.055f, 1.4f)*2.4f*1/1.055f : 1/12.92) * 100.0;
        gB = ((B > 0.04045) ? pow((B + 0.055f) / 1.055f, 1.4f)*2.4f*1/1.055f : 1/12.92) * 100.0;

        R = ((R > 0.04045) ? pow((R + 0.055f) / 1.055f, 2.4f) : (R / 12.92)) * 100.0;
        G = ((G > 0.04045) ? pow((G + 0.055f) / 1.055f, 2.4f) : (G / 12.92)) * 100.0;
        B = ((B > 0.04045) ? pow((B + 0.055f) / 1.055f, 2.4f) : (B / 12.92)) * 100.0;
        
        x = R*0.4124564 + G*0.3575761 + B*0.1804375;
        y = R*0.2126729 + G*0.7151522 + B*0.0721750;
        z = R*0.0193339 + G*0.1191920 + B*0.9503041;

        x = x / 95.047;
        y = y / 100.00;
        z = z / 108.883;
        
        gx = (x > 0.008856) ? 1/3.f*pow(float(x+1e-6),-2.f/3.f) : 7.787;
        gy = (y > 0.008856) ? 1/3.f*pow(float(y+1e-6),-2.f/3.f) : 7.787;
        gz = (z > 0.008856) ? 1/3.f*pow(float(z+1e-6),-2.f/3.f) : 7.787;
        
        float g_xr = gx * 1/95.047 * 0.4124564 * gR;
        float g_xg = gx * 1/95.047 * 0.3575761 * gG;
        float g_xb = gx * 1/95.047 * 0.1804375 * gB;

        float g_yr = gy * 1/100.00 * 0.2126729 * gR;
        float g_yg = gy * 1/100.00 * 0.7151522 * gG;
        float g_yb = gy * 1/100.00 * 0.0721750 * gB;

        float g_zr = gz * 1/108.883 * 0.0193339 * gR;
        float g_zg = gz * 1/108.883 * 0.1191920 * gG;
        float g_zb = gz * 1/108.883 * 0.9503041 * gB;

        atomicAdd(&image_grad[b][h][w][0], output_grad[b][h][w][0] * 
            116.f * g_yr);
        atomicAdd(&image_grad[b][h][w][0], output_grad[b][h][w][1] * 
            500.f * (g_xr - g_yr));
        atomicAdd(&image_grad[b][h][w][0], output_grad[b][h][w][2] * 
            200.f * (g_yr - g_zr));

        atomicAdd(&image_grad[b][h][w][1], output_grad[b][h][w][0] * 
            116.f * g_yg);
        atomicAdd(&image_grad[b][h][w][1], output_grad[b][h][w][1] * 
            500.f * (g_xg - g_yg));
        atomicAdd(&image_grad[b][h][w][1], output_grad[b][h][w][2] * 
            200.f * (g_yg - g_zg));

        atomicAdd(&image_grad[b][h][w][2], output_grad[b][h][w][0] * 
            116.f * g_yb);
        atomicAdd(&image_grad[b][h][w][2], output_grad[b][h][w][1] * 
            500.f * (g_xb - g_yb));
        atomicAdd(&image_grad[b][h][w][2], output_grad[b][h][w][2] * 
            200.f * (g_yb - g_zb));

        }
    }

    template <typename scalar_t>
    __global__ void LAB2RGB_sRGB_Forward(
        const int nthreads, 
        const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image, 
        torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output, 
        const int batch, 
        const int height, 
        const int width, 
        const int channel) {

        CUDA_1D_KERNEL_LOOP(index, nthreads) {
        
        int b = index / (height * width);
        index %= height * width;
        int h = index / width;
        index %= width;
        int w = index;

        float L = image[b][h][w][0];
        float A = image[b][h][w][1];
        float Bl = image[b][h][w][2];
        float R, G, B;
        
        float y = (L + 16.0) / 116.0;
        float x = A / 500.0 + y;
        float z = y - Bl / 200.0;
        
        float x3 = x * x * x;
        float y3 = y * y * y;
        float z3 = z * z * z;
        
        x = ((x3 > 0.008856) ? x3 : ((x - 16.0 / 116.0) / 7.787)) * 95.047;
        y = ((y3 > 0.008856) ? y3 : ((y - 16.0 / 116.0) / 7.787)) * 100.0;
        z = ((z3 > 0.008856) ? z3 : ((z - 16.0 / 116.0) / 7.787)) * 108.883;

        x = x / 100.0;
        y = y / 100.0;
        z = z / 100.0;
        
        R = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
        G = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
        B = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;
        
        R = ((R > 0.0031308) ? (1.055*pow(R, 1 / 2.4f) - 0.055) : (12.92*R));
        G = ((G > 0.0031308) ? (1.055*pow(G, 1 / 2.4f) - 0.055) : (12.92*G));
        B = ((B > 0.0031308) ? (1.055*pow(B, 1 / 2.4f) - 0.055) : (12.92*B));
        
        output[b][h][w][0] = R;
        output[b][h][w][1] = G;
        output[b][h][w][2] = B;
        }
    }

    template <typename scalar_t>
    __global__ void LAB2RGB_sRGB_Backward(
        const int nthreads, 
        const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output_grad, 
        const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image, 
        torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image_grad, 
        const int batch, 
        const int height, 
        const int width, 
        const int channel) {

        CUDA_1D_KERNEL_LOOP(index, nthreads) {
        
        int b = index / (height * width);
        index %= height * width;
        int h = index / width;
        index %= width;
        int w = index;

        float L = image[b][h][w][0];
        float A = image[b][h][w][1];
        float Bl = image[b][h][w][2];
        float R, G, B;
        
        float y = (L + 16.0) / 116.0;
        float x = A / 500.0 + y;
        float z = y - Bl / 200.0;
        
        float x3 = x * x * x;
        float y3 = y * y * y;
        float z3 = z * z * z;
        
        float gx, gy, gz, gR, gG, gB;
        gx = ((x3 > 0.008856) ? 3*x*x : 1/7.787) * 95.047;
        gy = ((y3 > 0.008856) ? 3*y*y : 1/7.787) * 100.0;
        gz = ((z3 > 0.008856) ? 3*z*z : 1/7.787) * 108.883;
        x = ((x3 > 0.008856) ? x3 : ((x - 16.0 / 116.0) / 7.787)) * 95.047;
        y = ((y3 > 0.008856) ? y3 : ((y - 16.0 / 116.0) / 7.787)) * 100.0;
        z = ((z3 > 0.008856) ? z3 : ((z - 16.0 / 116.0) / 7.787)) * 108.883;

        x = x / 100.0;
        y = y / 100.0;
        z = z / 100.0;
        
        R = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
        G = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
        B = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;
        
        gR = ((R > 0.0031308) ? (1.055*1/2.4f*pow(float(R+1e-6), 1/2.4f-1)) : 12.92); 
        gG = ((G > 0.0031308) ? (1.055*1/2.4f*pow(float(G+1e-6), 1/2.4f-1)) : 12.92); 
        gB = ((B > 0.0031308) ? (1.055*1/2.4f*pow(float(B+1e-6), 1/2.4f-1)) : 12.92); 
        
        float g_yl = 1/100.0 * gy * 1/116.0;
        float g_xl = 1/100.0 * gx * g_yl;
        float g_zl = 1/100.0 * gz * g_yl;

        float g_xa = 1/100.0 * gx * 1/500.0;

        float g_zb = 1/100.0 * gz * (-1/200.0);

        atomicAdd(&image_grad[b][h][w][0], output_grad[b][h][w][0] * 
            gR * (3.2404542f*g_xl+(-1.5371385f)*g_yl+(-0.4985314f)*g_zl));
        atomicAdd(&image_grad[b][h][w][0], output_grad[b][h][w][1] * 
            gG * (-0.9692660f*g_xl+1.8760108f*g_yl+0.0415560f*g_zl));
        atomicAdd(&image_grad[b][h][w][0], output_grad[b][h][w][2] * 
            gB * (0.0556434f*g_xl+(-0.2040259f)*g_yl+1.0572252f*g_zl));

        atomicAdd(&image_grad[b][h][w][1], output_grad[b][h][w][0] * 
            gR * (3.2404542f*g_xa));
        atomicAdd(&image_grad[b][h][w][1], output_grad[b][h][w][1] * 
            gG * (-0.9692660f)*g_xa);
        atomicAdd(&image_grad[b][h][w][1], output_grad[b][h][w][2] * 
            gB * 0.0556434f*g_xa);

        atomicAdd(&image_grad[b][h][w][2], output_grad[b][h][w][0] * 
            gR * (-0.4985314f)*g_zb);
        atomicAdd(&image_grad[b][h][w][2], output_grad[b][h][w][1] * 
            gG * 0.0415560f*g_zb);
        atomicAdd(&image_grad[b][h][w][2], output_grad[b][h][w][2] * 
            gB * 1.0572252f*g_zb);
        }
    }
} // namespace

    torch::Tensor RGB2LAB_sRGB_ForwardLaucher(
        torch::Tensor image 
        ) {

        const int batch = image.size(0);
        const int height = image.size(1);
        const int width = image.size(2);
        const int channel = image.size(3);

        auto output = torch::zeros_like(image);

        const int kThreadsPerBlock = 1024;
        const int output_size = batch * height * width;

        AT_DISPATCH_FLOATING_TYPES(image.type(), "RGB2LAB_sRGB_Forward", ([&] {
            RGB2LAB_sRGB_Forward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
                output_size, 
                image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
                output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
                batch, height, width, channel);
        }));

        
        return output;
    }

    torch::Tensor RGB2LAB_sRGB_BackwardLaucher(
        torch::Tensor output_grad, 
        torch::Tensor image 
        ) {

        const int batch = image.size(0);
        const int height = image.size(1);
        const int width = image.size(2);
        const int channel = image.size(3);

        auto image_grad = torch::zeros_like(image);

        const int kThreadsPerBlock = 1024;
        const int output_size = batch * height * width;

        AT_DISPATCH_FLOATING_TYPES(image.type(), "RGB2LAB_sRGB_Backward", ([&] {
            RGB2LAB_sRGB_Backward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
                output_size, 
                output_grad.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
                image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
                image_grad.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
                batch, height, width, channel);
        }));

        
        return image_grad;
    }

    torch::Tensor LAB2RGB_sRGB_ForwardLaucher(
        torch::Tensor image 
        ) {

        const int batch = image.size(0);
        const int height = image.size(1);
        const int width = image.size(2);
        const int channel = image.size(3);

        auto output = torch::zeros_like(image);

        const int kThreadsPerBlock = 1024;
        const int output_size = batch * height * width;

        AT_DISPATCH_FLOATING_TYPES(image.type(), "LAB2RGB_sRGB_Forward", ([&] {
            LAB2RGB_sRGB_Forward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
                output_size, 
                image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
                output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
                batch, height, width, channel);
        }));
        
        return output;
    }

    torch::Tensor LAB2RGB_sRGB_BackwardLaucher(
        torch::Tensor output_grad, 
        torch::Tensor image 
        ) {

        const int batch = image.size(0);
        const int height = image.size(1);
        const int width = image.size(2);
        const int channel = image.size(3);

        auto image_grad = torch::zeros_like(image);

        const int kThreadsPerBlock = 1024;
        const int output_size = batch * height * width;

        AT_DISPATCH_FLOATING_TYPES(image.type(), "LAB2RGB_sRGB_Backward", ([&] {
            LAB2RGB_sRGB_Backward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
                output_size, 
                output_grad.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
                image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
                image_grad.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(), 
                batch, height, width, channel);
        }));
        
        return image_grad;
    }

