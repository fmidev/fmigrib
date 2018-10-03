#include "NFmiGribPacking.h"
#include <cassert>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

void NFmiGribPacking::UnpackBitmap(const unsigned char* __restrict__ bitmap, int* __restrict__ unpacked, size_t len,
                                   size_t unpackedLen)
{
	size_t i, idx = 0;
	int v = 1;

	short j = 0;

	for (i = 0; i < len; i++)
	{
		for (j = 7; j >= 0; j--)
		{
			if (BitTest(bitmap[i], j))
			{
				unpacked[idx] = v++;
			}
			else
			{
				unpacked[idx] = 0;
			}

			if (++idx >= unpackedLen)
			{
				// packed data might not be aligned nicely along byte boundaries --
				// need to break from loop after final element has been processed
				break;
			}
		}
	}
}

template <typename T>
__host__ T Min(T* d_arr, size_t N, cudaStream_t& stream)
{
	T* ret = thrust::min_element(thrust::cuda::par.on(stream), d_arr, d_arr + N);

	return *ret;
}

template <typename T>
__host__ T Max(T* d_arr, size_t N, cudaStream_t& stream)
{
	T* ret = thrust::max_element(thrust::cuda::par.on(stream), d_arr, d_arr + N);

	return *ret;
}

__device__ void SetBitOn(unsigned char* p, long bitp)
{
	p += bitp / 8;
	*p |= (1u << (7 - ((bitp) % 8)));
}

__device__ void SetBitOff(unsigned char* p, long bitp)
{
	p += bitp / 8;
	*p &= ~(1u << (7 - ((bitp) % 8)));
}

long NFmiGribPacking::simple_packing::get_binary_scale_fact(double max, double min, long bpval)
{
	assert(max >= min);
	double range = max - min;
	double zs = 1;
	long scale = 0;
	const long last = 127; /* Depends on edition, should be parameter */

	unsigned long maxint = NFmiGribPacking::ToPower(bpval, 2) - 1;
	double dmaxint = (double)maxint;

	assert(bpval >= 1);

	if (range == 0)
		return 0;

	/* range -= 1e-10; */
	while ((range * zs) <= dmaxint)
	{
		scale--;
		zs *= 2;
	}

	while ((range * zs) > dmaxint)
	{
		scale++;
		zs /= 2;
	}

	while ((unsigned long)(range * zs + 0.5) <= maxint)
	{
		scale--;
		zs *= 2;
	}

	while ((unsigned long)(range * zs + 0.5) > maxint)
	{
		scale++;
		zs /= 2;
	}

	if (scale < -last)
	{
		printf("grib_get_binary_scale_fact: max=%g min=%g\n", max, min);
		scale = -last;
	}
	assert(scale <= last);

	return scale;
}

long NFmiGribPacking::simple_packing::get_decimal_scale_fact(double max, double min, long bpval, long binary_scale)
{
	assert(max >= min);

	double range = max - min;
	const long last = 127; /* Depends on edition, should be parameter */
	double decimal_scale_factor = 0;
	double f;
	double minrange = 0, maxrange = 0;
	double decimal = 1;
	long bits_per_value = bpval;

	double unscaled_min = min;
	double unscaled_max = max;

	f = NFmiGribPacking::ToPower(bits_per_value, 2) - 1;
	minrange = NFmiGribPacking::ToPower(-last, 2) * f;
	maxrange = NFmiGribPacking::ToPower(last, 2) * f;

	while (range < minrange)
	{
		decimal_scale_factor += 1;
		decimal *= 10;
		min = unscaled_min * decimal;
		max = unscaled_max * decimal;
		range = (max - min);
	}
	while (range > maxrange)
	{
		decimal_scale_factor -= 1;
		decimal /= 10;
		min = unscaled_min * decimal;
		max = unscaled_max * decimal;
		range = (max - min);
	}

	return decimal_scale_factor;
}

__device__ void PackUnevenBytes(unsigned char* __restrict__ d_p, const double* __restrict__ d_u, size_t values_len,
                                NFmiGribPacking::packing_coefficients coeff, int idx)
{
	double decimal = NFmiGribPacking::ToPower(-coeff.decimalScaleFactor, 10);
	double divisor = NFmiGribPacking::ToPower(-coeff.binaryScaleFactor, 2);

	double x = (((d_u[idx] * decimal) - coeff.referenceValue) * divisor) + 0.5;
	unsigned long unsigned_val = static_cast<unsigned long>(x);

	long bitp = coeff.bitsPerValue * idx;

	long i = 0;

	for (i = coeff.bitsPerValue - 1; i >= 0; i--)
	{
		if (BitTest(unsigned_val, i))
		{
			SetBitOn(d_p, bitp);
		}
		else
		{
			SetBitOff(d_p, bitp);
		}

		bitp++;
	}
}

__device__ void PackFullBytes(unsigned char* __restrict__ d_p, const double* __restrict__ d_u, size_t values_len,
                              NFmiGribPacking::packing_coefficients coeff, int idx)
{
	double decimal = NFmiGribPacking::ToPower(-coeff.decimalScaleFactor, 10);
	double divisor = NFmiGribPacking::ToPower(-coeff.binaryScaleFactor, 2);

	double x = ((((d_u[idx] * decimal) - coeff.referenceValue) * divisor) + 0.5);
	unsigned long unsigned_val = static_cast<unsigned long>(x);

	unsigned char* encoded = &d_p[idx * static_cast<int>(coeff.bitsPerValue / 8)];

	while (coeff.bitsPerValue >= 8)
	{
		coeff.bitsPerValue -= 8;
		*encoded = (unsigned_val >> coeff.bitsPerValue);
		encoded++;
	}
}

__global__ void PackSimpleKernel(const double* d_u, unsigned char* d_p, const int* d_b, size_t N,
                                 NFmiGribPacking::packing_coefficients coeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (coeff.bitsPerValue %
		    8)  // modulo is expensive but "Compiler will convert literal power-of-2 divides to bitwise shifts"
		{
			PackUnevenBytes(d_p, d_u, N, coeff, idx);
		}
		else
		{
			PackFullBytes(d_p, d_u, N, coeff, idx);
		}
	}
}

bool NFmiGribPacking::simple_packing::Pack(double* arr, unsigned char* packed, const int* d_bitmap, size_t unpackedLen,
                                           NFmiGribPacking::packing_coefficients coeffs, cudaStream_t& stream)
{
	// 1. Check pointer type

	bool isHostMemory = IsHostPointer(arr);

	double* d_arr = 0;

	if (!isHostMemory)
	{
		d_arr = arr;
	}

	// 2. Copy unpacked data to device if needed

	if (isHostMemory)
	{
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_arr), unpackedLen * sizeof(double)));
		CUDA_CHECK(cudaMemcpyAsync(d_arr, reinterpret_cast<void*>(arr), unpackedLen * sizeof(double),
		                           cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaStreamSynchronize(stream));
	}

	unsigned char* d_packed = 0;

	long packedLen = ((coeffs.bitsPerValue * unpackedLen) + 7) / 8;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_packed), packedLen * sizeof(unsigned char)));

	int blockSize = 512;
	int gridSize = unpackedLen / blockSize + (unpackedLen % blockSize == 0 ? 0 : 1);

	PackSimpleKernel<<<gridSize, blockSize, 0, stream>>>(d_arr, d_packed, d_bitmap, unpackedLen, coeffs);

	CUDA_CHECK(cudaMemcpyAsync(packed, d_packed, packedLen * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_packed));

	return true;
}
