/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};
struct offsetsAdjacencyPair{
	int offset_x;
	int offset_y;
	uint64_t one_hot_adjacency; 
};
// uint64_t unit_num = 1;
__device__ const offsetsAdjacencyPair offsetsAdjacencyPairs[] = {
	// {-1,  0, (unit_num << 0)},
	// { 0,  1, (unit_num << 1)},
	// { 1,  0, (unit_num << 2)},
	// { 0, -1, (unit_num << 3)}, // level-1 over 4
	// {-2,  0, (unit_num << 4)},
	// {-1,  1, (unit_num << 5)},
	// { 0,  2, (unit_num << 6)},
	// { 1,  1, (unit_num << 7)},
	// { 2,  0, (unit_num << 8)},
	// { 1, -1, (unit_num << 9)},
	// { 0, -2, (unit_num << 10)},
	// {-1, -1, (unit_num << 11)}, // level-2 over 8
	// {-3,  0, (unit_num << 12)},
	// {-2,  1, (unit_num << 13)},
	// {-1,  2, (unit_num << 14)},
	// { 0,  3, (unit_num << 15)},
	// { 1,  2, (unit_num << 16)},
	// { 2,  1, (unit_num << 17)},
	// { 3,  0, (unit_num << 18)},
	// { 2, -1, (unit_num << 19)},
	// { 1, -2, (unit_num << 20)},
	// { 0, -3, (unit_num << 21)},
	// {-1, -2, (unit_num << 22)},
	// {-2, -1, (unit_num << 23)}, // level-3 over 12
	// {-4,  0, (unit_num << 24)},
	// {-3,  1, (unit_num << 25)},
	// {-2,  2, (unit_num << 26)},
	// {-1,  3, (unit_num << 27)},
	// { 0,  4, (unit_num << 28)},
	// { 1,  3, (unit_num << 29)},
	// { 2,  2, (unit_num << 30)},
	// { 3,  1, (unit_num << 31)},
	// { 4,  0, (unit_num << 32)},
	// { 3, -1, (unit_num << 33)},
	// { 2, -2, (unit_num << 34)},
	// { 1, -3, (unit_num << 35)},
	// { 0, -4, (unit_num << 36)},
	// {-1, -3, (unit_num << 37)},
	// {-2, -2, (unit_num << 38)},
	// {-3, -1, (unit_num << 39)}, // level-4 over 16
	// {-5,  0, (unit_num << 40)},
	// {-4,  1, (unit_num << 41)},
	// {-3,  2, (unit_num << 42)},
	// {-2,  3, (unit_num << 43)},
	// {-1,  4, (unit_num << 44)},
	// { 0,  5, (unit_num << 45)},
	// { 1,  4, (unit_num << 46)},
	// { 2,  3, (unit_num << 47)},
	// { 3,  2, (unit_num << 48)},
	// { 4,  1, (unit_num << 49)},
	// { 5,  0, (unit_num << 50)},
	// { 4, -1, (unit_num << 51)},
	// { 3, -2, (unit_num << 52)},
	// { 2, -3, (unit_num << 53)},
	// { 1, -4, (unit_num << 54)},
	// { 0, -5, (unit_num << 55)}, 
	// {-1, -4, (unit_num << 56)},
	// {-2, -3, (unit_num << 57)},
	// {-3, -2, (unit_num << 58)},
	// {-4, -1, (unit_num << 59)}, // level-5 over 20
	{-1,  0, 0x0000000000000001},
	{ 0,  1, 0x0000000000000002},
	{ 1,  0, 0x0000000000000004},
	{ 0, -1, 0x0000000000000008}, // level-1 over 4
	{-2,  0, 0x0000000000000010},
	{-1,  1, 0x0000000000000020},
	{ 0,  2, 0x0000000000000040},
	{ 1,  1, 0x0000000000000080},
	{ 2,  0, 0x0000000000000100},
	{ 1, -1, 0x0000000000000200},
	{ 0, -2, 0x0000000000000400},
	{-1, -1, 0x0000000000000800}, // level-2 over 8
	{-3,  0, 0x0000000000001000},
	{-2,  1, 0x0000000000002000},
	{-1,  2, 0x0000000000004000},
	{ 0,  3, 0x0000000000008000},
	{ 1,  2, 0x0000000000010000},
	{ 2,  1, 0x0000000000020000},
	{ 3,  0, 0x0000000000040000},
	{ 2, -1, 0x0000000000080000},
	{ 1, -2, 0x0000000000100000},
	{ 0, -3, 0x0000000000200000},
	{-1, -2, 0x0000000000400000},
	{-2, -1, 0x0000000000800000}, // level-3 over 12
	{-4,  0, 0x0000000001000000},
	{-3,  1, 0x0000000002000000},
	{-2,  2, 0x0000000004000000},
	{-1,  3, 0x0000000008000000},
	{ 0,  4, 0x0000000010000000},
	{ 1,  3, 0x0000000020000000},
	{ 2,  2, 0x0000000040000000},
	{ 3,  1, 0x0000000080000000},
	{ 4,  0, 0x0000000100000000},
	{ 3, -1, 0x0000000200000000},
	{ 2, -2, 0x0000000400000000},
	{ 1, -3, 0x0000000800000000},
	{ 0, -4, 0x0000001000000000},
	{-1, -3, 0x0000002000000000},
	{-2, -2, 0x0000004000000000},
	{-3, -1, 0x0000008000000000}, // level-4 over 16
	{-5,  0, 0x0000010000000000},
	{-4,  1, 0x0000020000000000},
	{-3,  2, 0x0000040000000000},
	{-2,  3, 0x0000080000000000},
	{-1,  4, 0x0000100000000000},
	{ 0,  5, 0x0000200000000000},
	{ 1,  4, 0x0000400000000000},
	{ 2,  3, 0x0000800000000000},
	{ 3,  2, 0x0001000000000000},
	{ 4,  1, 0x0002000000000000},
	{ 5,  0, 0x0004000000000000},
	{ 4, -1, 0x0008000000000000},
	{ 3, -2, 0x0010000000000000},
	{ 2, -3, 0x0020000000000000},
	{ 1, -4, 0x0040000000000000},
	{ 0, -5, 0x0080000000000000}, 
	{-1, -4, 0x0100000000000000},
	{-2, -3, 0x0200000000000000},
	{-3, -2, 0x0400000000000000},
	{-4, -1, 0x0800000000000000}, // level-5 over 20
};
__forceinline__ __device__ uint64_t getAdjacencyCode(int offset_x, int offset_y)
{
	// for (int i = 0; i < sizeof(offsetsAdjacencyPair)/sizeof(offsetsAdjacencyPair[0]); i++)
	for (int i = 0; i < 60; i++) // hardcoded length (4+8+12+16+20)
	{
		if (offsetsAdjacencyPairs[i].offset_x == offset_x && offsetsAdjacencyPairs[i].offset_y == offset_y)
			return offsetsAdjacencyPairs[i].one_hot_adjacency;
	}
}
__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float2 vecSubtract(const float2 &a, const float2 &b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}


__forceinline__ __device__ float vecDot(const float2 &a, const float2 &b)
{
    return a.x * b.x + a.y * b.y;
}

__forceinline__ __device__ float vecLength(const float2 &v)
{
    return sqrtf(v.x * v.x + v.y * v.y);
}

__forceinline__ __device__ float2 vecNormalize(const float2 &v)
{
    float len = vecLength(v);
    if (len > 0)
        return make_float2(v.x / len, v.y / len);
    else
        return make_float2(0.0f, 0.0f);
}

__forceinline__ __device__ float2 vecPerpendicular(const float2 &v)
{
    return make_float2(-v.y, v.x);
}
__forceinline__ __device__ bool SAT(float2 *rect1, float2 *rect2)
{
	float2 axes[4];

    float2 edge1_0 = vecSubtract(rect1[1], rect1[0]);
    float2 edge1_1 = vecSubtract(rect1[2], rect1[1]);

    axes[0] = vecNormalize(vecPerpendicular(edge1_0));
    axes[1] = vecNormalize(vecPerpendicular(edge1_1));

    float2 edge2_0 = vecSubtract(rect2[1], rect2[0]);
    float2 edge2_1 = vecSubtract(rect2[2], rect2[1]);

    axes[2] = vecNormalize(vecPerpendicular(edge2_0));
    axes[3] = vecNormalize(vecPerpendicular(edge2_1));

    for (int i = 0; i < 4; i++)
    {
        float2 axis = axes[i];

        float min1 = vecDot(rect1[0], axis);
        float max1 = min1;
        for (int j = 1; j < 4; j++)
        {
            float projection = vecDot(rect1[j], axis);
            if (projection < min1)
                min1 = projection;
            else if (projection > max1)
                max1 = projection;
        }
        float min2 = vecDot(rect2[0], axis);
        float max2 = min2;
        for (int j = 1; j < 4; j++)
        {
            float projection = vecDot(rect2[j], axis);
            if (projection < min2)
                min2 = projection;
            else if (projection > max2)
                max2 = projection;
        }

        if (max1 < min2 || max2 < min1)
        {
            return false;
        }
    }
    return true;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ void getRect(const float2 p, int2 ext_rect, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + ext_rect.x + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y - 1) / BLOCK_Y)))
	};
}


__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
