#include "NFmiGribPacking.h"
#include <cassert>

const double kFloatMissing = 32700.;

__device__
void GetBitValue(const unsigned char* p, long bitp, int *val)
{
	p += (bitp >> 3);
	*val = (*p&(1<<(7-(bitp%8))));
}

__device__
void UnpackFullBytes(const unsigned char* __restrict__ d_p, double* __restrict__ d_u, const int* __restrict__ d_b, NFmiGribPacking::packing_coefficients coeff, int idx)
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
			d_u[idx] = kFloatMissing;
			value_found = 0;
		}
		else
		{
			bm--;
		}
	}

	if (value_found)
	{
		size_t o = bm*l;

		lvalue	= 0;
		lvalue	<<= 8;
		lvalue |= d_p[o++] ;

		for ( bc=1; bc<l; bc++ )
		{
			lvalue <<= 8;
			lvalue |= d_p[o++] ;
		}

		d_u[idx] = ((lvalue * coeff.binaryScaleFactor) + coeff.referenceValue) * coeff.decimalScaleFactor;
	}
}

__device__
void UnpackUnevenBytes(const unsigned char* __restrict__ d_p, double* __restrict__ d_u, const int* __restrict__ d_b, NFmiGribPacking::packing_coefficients coeff, int idx)
{
	int j=0;
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
			d_u[idx] = kFloatMissing;
			value_found = 0;
		}
		else
		{
			bm--;
		}
	}

	if (value_found)
	{
		long bitp=coeff.bitsPerValue*bm;

		lvalue=0;

		for(j=0; j< coeff.bitsPerValue; j++)
		{
			lvalue <<= 1;
			int val;

			GetBitValue(d_p, bitp, &val);

			if (val) lvalue += 1;

			bitp += 1;
		}

		d_u[idx] = ((lvalue * coeff.binaryScaleFactor) + coeff.referenceValue) * coeff.decimalScaleFactor;
	}

}

__global__
void UnpackSimpleKernel(double* d_u, const unsigned char* d_p, const int* d_b, size_t N, NFmiGribPacking::packing_coefficients coeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (coeff.bitsPerValue % 8) // modulo is expensive but "Compiler will convert literal power-of-2 divides to bitwise shifts"
		{
			UnpackUnevenBytes(d_p, d_u, d_b, coeff, idx);
		}
		else
		{
			UnpackFullBytes(d_p, d_u, d_b, coeff, idx);
		}
	}
}

bool NFmiGribPacking::simple_packing::Unpack(double* arr, const unsigned char* packed, const int* d_bitmap, size_t unpackedLen, size_t packedLen, NFmiGribPacking::packing_coefficients coeffs, cudaStream_t& stream)
{

	bool isHostMemory = IsHostPointer(arr);

	double* d_arr = 0;

	if (isHostMemory)
	{
		CUDA_CHECK(cudaMalloc(&d_arr, unpackedLen * sizeof(double)));
		CUDA_CHECK(cudaMemcpyAsync(d_arr, arr, unpackedLen * sizeof(double), cudaMemcpyHostToDevice, stream));
	}
	else
	{
		d_arr = arr;
	}
	
	assert(packedLen > 0);
	assert(d_arr);
	assert(packed);

	unsigned char* d_packed = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**> (&d_packed), packedLen * sizeof(unsigned char)));
	CUDA_CHECK(cudaMemcpyAsync(d_packed, packed, packedLen * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));
	
	int blockSize = 512;
	int gridSize = unpackedLen / blockSize + (unpackedLen % blockSize == 0 ? 0 : 1);

	UnpackSimpleKernel <<< gridSize, blockSize, 0, stream >>> (d_arr, d_packed, d_bitmap, unpackedLen, coeffs);

	if (isHostMemory)
	{
		CUDA_CHECK(cudaMemcpyAsync(arr, d_arr, sizeof(double) * unpackedLen, cudaMemcpyDeviceToHost, stream));
    	CUDA_CHECK(cudaFree(d_arr));
	}
	
	CUDA_CHECK(cudaFree(d_packed));

	return true;

}
