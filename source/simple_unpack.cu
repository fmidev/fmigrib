#include "NFmiGribPacking.h"
#include <cassert>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

__device__ void GetBitValue(const unsigned char* p, long bitp, int* val)
{
	p += (bitp >> 3);
	*val = (*p & (1 << (7 - (bitp % 8))));
}

template <typename T>
__global__ void CopyWithBitmap(T* d_arr, int* d_b, T value, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (d_b[idx] == 1)
		{
			d_arr[idx] = value;
		}
	}
}

template <typename T>
__device__ void UnpackFullBytes(const unsigned char* __restrict__ d_p, T* __restrict__ d_u, const int* __restrict__ d_b,
                                NFmiGribPacking::packing_coefficients coeff, int idx)
{
	int bc;
	unsigned long lvalue;

	int l = coeff.bitsPerValue / 8;

	int bm = idx;
	int value_found = 1;

	if (d_b)
	{
		bm = d_b[idx];

		if (bm == 0)
		{
			d_u[idx] = NFmiGribPacking::simple_packing::MissingValue<T>();
			value_found = 0;
		}
		else
		{
			bm--;
		}
	}

	if (value_found)
	{
		size_t o = bm * l;

		lvalue = 0;
		lvalue <<= 8;
		lvalue |= d_p[o++];

		for (bc = 1; bc < l; bc++)
		{
			lvalue <<= 8;
			lvalue |= d_p[o++];
		}

		d_u[idx] = fma(lvalue, coeff.binaryScaleFactor, coeff.referenceValue) * coeff.decimalScaleFactor;
	}
}

template <typename T>
__device__ void UnpackUnevenBytes(const unsigned char* __restrict__ d_p, T* __restrict__ d_u,
                                  const int* __restrict__ d_b, NFmiGribPacking::packing_coefficients coeff, int idx)
{
	int j = 0;
	unsigned long lvalue;

	int bm = idx;
	int value_found = 1;

	/*
	 * Check if bitmap is set.
	 * If bitmap is set and indicates that value for this element is missing, do
	 * not proceed to calculating phase.
	 *
	 * If bitmap is set and indicates that value exists for this element, the index
	 * for the actual data is the one indicated by the bitmap array. From this index
	 * we reduce one (1) because that one is added to the value in unpack_bitmap.
	 */

	if (d_b)
	{
		bm = d_b[idx];

		if (bm == 0)
		{
			d_u[idx] = NFmiGribPacking::simple_packing::MissingValue<T>();
			value_found = 0;
		}
		else
		{
			bm--;
		}
	}

	if (value_found)
	{
		long bitp = coeff.bitsPerValue * bm;

		lvalue = 0;

		for (j = 0; j < coeff.bitsPerValue; j++)
		{
			lvalue <<= 1;
			int val;

			GetBitValue(d_p, bitp, &val);

			if (val)
				lvalue += 1;

			bitp += 1;
		}

		d_u[idx] = fma(lvalue, coeff.binaryScaleFactor, coeff.referenceValue) * coeff.decimalScaleFactor;
	}
}

template <typename T>
__global__ void UnpackSimpleKernel(const unsigned char* d_p, T* d_u, const int* d_b, size_t N,
                                   NFmiGribPacking::packing_coefficients coeff)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (coeff.bitsPerValue % 8)
		{
			UnpackUnevenBytes(d_p, d_u, d_b, coeff, idx);
		}
		else
		{
			UnpackFullBytes(d_p, d_u, d_b, coeff, idx);
		}
	}
}

template <typename T>
__host__ bool NFmiGribPacking::simple_packing::Unpack(T* arr, const unsigned char* packed, const int* d_bitmap,
                                                      size_t unpackedLen, size_t packedLen,
                                                      NFmiGribPacking::packing_coefficients coeffs,
                                                      cudaStream_t& stream)
{
	const bool isHostMemory = IsHostPointer(arr);

	// Destination data array allocation
	T* d_arr = 0;

	if (isHostMemory)
	{
		CUDA_CHECK(cudaMalloc(&d_arr, unpackedLen * sizeof(T)));
		CUDA_CHECK(cudaMemcpyAsync(d_arr, arr, unpackedLen * sizeof(T), cudaMemcpyHostToDevice, stream));
	}
	else
	{
		d_arr = arr;
	}

	// Packed data array allocation

	unsigned char* d_packed = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_packed), packedLen * sizeof(unsigned char)));
	CUDA_CHECK(cudaMemcpyAsync(d_packed, packed, packedLen * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));

	// Bitmap data array allocation

	int* d_b = 0;
	if (d_bitmap)
	{
		CUDA_CHECK(cudaMalloc((void**)(&d_b), unpackedLen * sizeof(int)));
		CUDA_CHECK(cudaMemcpyAsync(d_b, d_bitmap, unpackedLen * sizeof(int), cudaMemcpyHostToDevice, stream));
	}

	const int blockSize = 512;
	const int gridSize = unpackedLen / blockSize + (unpackedLen % blockSize == 0 ? 0 : 1);

	UnpackSimpleKernel<T><<<gridSize, blockSize, 0, stream>>>(d_packed, d_arr, d_b, unpackedLen, coeffs);

	if (isHostMemory)
	{
		CUDA_CHECK(cudaMemcpyAsync(arr, d_arr, sizeof(T) * unpackedLen, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaFree(d_arr));
	}

	CUDA_CHECK(cudaFree(d_packed));

	if (d_b)
	{
		CUDA_CHECK(cudaFree(d_b));
	}

	return true;
}

template __host__ bool NFmiGribPacking::simple_packing::Unpack<double>(double*, const unsigned char*, const int*,
                                                                       size_t, size_t,
                                                                       NFmiGribPacking::packing_coefficients,
                                                                       cudaStream_t&);
template __host__ bool NFmiGribPacking::simple_packing::Unpack<float>(float*, const unsigned char*, const int*, size_t,
                                                                      size_t, NFmiGribPacking::packing_coefficients,
                                                                      cudaStream_t&);
