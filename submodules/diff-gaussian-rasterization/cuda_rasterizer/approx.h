#ifndef APPROX_H_
#define APPROX_H_

#include <cuda_fp16.h>
#include <iostream>
#include <cmath>

#define __HOST_DEVICE__ __host__ __device__
#define DIV_TABLE_SIZE 1024
#define EXP_TABLE_SIZE 256 //1024
#define MSB_N 8 //10
#define LSB_N 8 //6
#define FRAC_N 12
#define QUAN_W 8

struct half2 {
    __half x, y;
    __HOST_DEVICE__ half2(): x(0), y(0) {}
    __HOST_DEVICE__ half2(__half x, __half y): x(x), y(y) {}
    __HOST_DEVICE__ half2& operator=(const float2& f2) {
        x = __float2half(f2.x);
        y = __float2half(f2.y);
        return *this;
    }
};

struct half4 {
    __half x, y, z, w;
    __HOST_DEVICE__ half4(): x(0), y(0), z(0), w(0) {}
    __HOST_DEVICE__ half4(__half x, __half y, __half z, __half w): x(x), y(y), z(z), w(w) {}
    __HOST_DEVICE__ half4& operator=(const float4& f4) {
        x = __float2half(f4.x);
        y = __float2half(f4.y);
        z = __float2half(f4.z);
        w = __float2half(f4.w);
        return *this;
    }
};

__forceinline__ __HOST_DEVICE__ uint32_t float2fix(float x_f, uint32_t frac_n) {
    return (uint32_t)floor(x_f * float(1 << frac_n));
}

__forceinline__ __HOST_DEVICE__ float fix2float(uint32_t x_u, uint32_t frac_n) {
    return (float)x_u / (float)(1 << frac_n);
}

__forceinline__ __device__ uint2 split_fix(uint32_t x, uint32_t msb_n, uint32_t lsb_n) {
    uint32_t lsb = x & ((1 << lsb_n) - 1);
    uint32_t msb = (x >> lsb_n) & ((1 << msb_n) - 1);
    uint2 r{msb, lsb};
    return r;
}

__forceinline__ __device__ __half exp_lut_approx(__half x_h, uint32_t frac_n, uint32_t msb_n, uint32_t lsb_n, const __half *exp_off_lut, const __half *exp_slope_lut) {
    // float x_f = __float2half(x_h);
    float x_f = __half2float(x_h);
    uint32_t x_u = float2fix(x_f, frac_n);
    uint2 x_seg = split_fix(x_u, msb_n, lsb_n);
    __half b = exp_off_lut[x_seg.x];
    __half k = exp_slope_lut[x_seg.x];
    __half dx = fix2float(x_seg.y, frac_n);
    return __hadd(b, __hmul(k, dx));
}

__forceinline__ __device__ __half div_lut_approx(__half x_h, uint32_t frac_n, uint32_t msb_n, uint32_t lsb_n, const __half *div_off_lut, const __half *div_slope_lut) {
    float x_f = __half2float(x_h);
    uint32_t x_u = float2fix(x_f, frac_n);
    uint2 x_seg = split_fix(x_u, msb_n, lsb_n);
    __half b = div_off_lut[x_seg.x];
    __half k = div_slope_lut[x_seg.x];
    __half dx = fix2float(x_seg.y, frac_n);
    return __hadd(b, __hmul(k, dx));
}

__forceinline__ __device__ float alpha_quan(float x_f, uint32_t quan_w) {
    float x_q = floor(x_f * float(1 << quan_w));
    float x_dq = x_q / float(1 << quan_w);
    return x_dq;
}

__forceinline__ __device__ __half half_alpha_quan(__half x_h, uint32_t quan_w) {
    float x_f = __half2float(x_h);

    float x_q = floor(x_f * float(1 << quan_w)); // floorf better
    float x_dq = x_q / float(1 << quan_w); // equals to / (1 << quan_w)
    return __float2half(x_dq);
}

__forceinline__ __device__ int fp_exp_bits_to_int(float x_f)
{
    unsigned int bits = __float_as_int(x_f);
    unsigned int exponent_val = (bits >> 23) & 0xFF;
    return static_cast<int>(exponent_val) - 127;
}

__forceinline__ __device__ int half_exp_bits_to_int(__half x_h)
{
    unsigned short bits = __half_as_ushort(x_h);
    unsigned int exponent_val = (bits >> 10) & 0x1F;
    return static_cast<int>(exponent_val) - 15;
}

#endif