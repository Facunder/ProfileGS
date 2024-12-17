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
#include <iostream>
// #include "pattern_pair_array.h"
#include "opacity_pattern_pair_array.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
// #define PATTERN_TYPES_NUM 3671 // 12 bit
// #define PATTERN_TYPES_NUM 14193 // 14 bit
#define PATTERN_TYPES_NUM 14315 // 14 bit
#define RADII_COPIES 16 // 4 bit
#define RADII_UNIT 4 // 4 bit
#define MAIN_DIRECTION_START_BIT 6 // 4 radii + 2 subloc
#define AXIS_RATIO_START_BIT 9 // 3 + 4 + 2
#define AXIS_RATIO_COPIES 4 // 2 bit
#define OPACITY_START_BIT 11
#define OPACITY_COPIES 8
#define PATTERN_BITS 40
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
__device__ const float angle_criteria[] = { // 3 bit 
	-2.4142f,
	-1.0000f,
	-0.4143f,
	0.0f,
	0.4142f,
	1.0000f,
	2.4142f
};
// __device__ const float angle_criteria[] = { // 4 bit
// 	-5.0273f,
//     -2.4142f,
//     -1.4966f,   
// 	-1.0000f,   
// 	-0.6682f,   
// 	-0.4142f,   
// 	-0.1989f,         
// 	0.0f,    
// 	0.1989f,
// 	0.4142f,
//     0.6682f,
//     1.0000f,
//     1.4966f,
// 	2.4142f,
//     5.0273f
// };
struct offsetsAdjacencyPair{
	int offset_x;
	int offset_y;
	uint64_t one_hot_adjacency; 
};
// uint64_t unit_num = 1;
__device__ const offsetsAdjacencyPair offsetsAdjacencyPairs[] = {
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
	{-4, -1, 0x0800000000000000} // level-5 over 20
};

__forceinline__ __device__ void getAngleFeature(uint32_t& feature, float tan_main_direction)
{
	for(int i = 0; i < 7; i++) {
		if(tan_main_direction < angle_criteria[i]) {
			feature |= (i << MAIN_DIRECTION_START_BIT);
			return;
		}
	}
	feature |= (7 << MAIN_DIRECTION_START_BIT);
}
__forceinline__ __device__ int popcount_uint64_t(uint64_t n) {
    unsigned int low = __popc(static_cast<unsigned int>(n)); 
	// printf("low: %d\n", low);
    unsigned int high = __popc(static_cast<unsigned int>(n >> 32)); 
	// printf("high: %d\n", high);
    return low + high;
}
__forceinline__ __device__ bool patternDecoder(uint64_t pattern, int indix, int& offset_x, int& offset_y) {
	if(pattern & offsetsAdjacencyPairs[indix].one_hot_adjacency){
		offset_x = offsetsAdjacencyPairs[indix].offset_x;
		offset_y = offsetsAdjacencyPairs[indix].offset_y;
		return true;
	} else 
		return false;
}
// __forceinline__ __device__ int patternMatchNum(uint32_t feature, bool& patterned) {
// 	for(int i = 0; i < 220578; i++){ // pattern length hardcoded
// 		if(feature == pattern_pairs[i].key){
// 			patterned = true;
// 			return popcount_uint64_t(pattern_pairs[i].value);
// 		}
// 	}
// 	patterned = false;
// 	return 0;
// }
// __forceinline__ __device__ uint64_t patternMatch(uint32_t feature) {
// 	for(int i = 0; i < 220578; i++){ // pattern length hardcoded
// 		if(feature == pattern_pairs[i].key){
// 			return pattern_pairs[i].value;
// 		}
// 	}
// }
__forceinline__ __device__ uint64_t patternMatch(uint32_t feature) {
    int left = 0;
    int right = PATTERN_TYPES_NUM - 1;

    while (left <= right) {
        int mid = (left + right) >> 1;
        uint32_t mid_key = pattern_pairs[mid].key;

        if (mid_key == feature) {
            return pattern_pairs[mid].value;
        } else if (mid_key < feature) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return 0ULL;
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
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
