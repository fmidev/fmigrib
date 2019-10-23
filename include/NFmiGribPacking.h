/*
 * File:   NFmiGribPacking.h
 *
 */

#ifndef NFMIGRIBPACKING_H
#define NFMIGRIBPACKING_H

#ifdef HAVE_CUDA

#include <iostream>

#if defined __GNUC__
#if __GNUC_MINOR__ > 5
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

#include <cuda_runtime.h>

#if defined __GNUC__ && __GNUC_MINOR__ > 5
#pragma GCC diagnostic pop
#endif

void CheckCudaError(cudaError_t errarg, const char* file, const int line);
void CheckCudaErrorString(const char* errstr, const char* file, const int line);

#define CUDA_CHECK(errarg) CheckCudaError(errarg, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR_MSG(errstr) CheckCudaErrorString(errstr, __FILE__, __LINE__)

inline void CheckCudaError(cudaError_t errarg, const char* file, const int line)
{
	if (errarg)
	{
		std::cerr << "Error at " << file << "(" << line << "): " << cudaGetErrorString(errarg) << std::endl;
		exit(1);
	}
}

inline void CheckCudaErrorString(const char* errstr, const char* file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << errstr << " " << file << " at (" << line << "): " << cudaGetErrorString(err)
		          << std::endl;

		exit(1);
	}
}

#define BitMask1(i) (1u << i)
#define BitTest(n, i) !!((n)&BitMask1(i))

namespace NFmiGribPacking
{
struct packing_coefficients
{
	int bitsPerValue;
	double binaryScaleFactor;
	double decimalScaleFactor;
	double referenceValue;

	packing_coefficients() : bitsPerValue(0), binaryScaleFactor(0), decimalScaleFactor(0), referenceValue(0)
	{
	}
};

namespace simple_packing
{
bool Unpack(double* d_arr, const unsigned char* d_packed, const int* d_bitmap, size_t unpackedLen, size_t packedLen,
            packing_coefficients coeffs, cudaStream_t& stream);
bool Pack(double* d_arr, unsigned char* d_packed, const int* d_bitmap, size_t unpackedLength,
          packing_coefficients coeffs, cudaStream_t& stream);
long get_decimal_scale_fact(double max, double min, long bpval, long binary_scale);
long get_binary_scale_fact(double max, double min, long bpval);
}

void UnpackBitmap(const unsigned char* __restrict__ bitmap, int* __restrict__ unpacked, size_t len, size_t unpackedLen);

__host__ __device__ double ToPower(double value, double power);
bool IsHostPointer(const double* ptr);
void MinMax(double* d, size_t unpackedLen, double& min, double& max, cudaStream_t& stream);
void Fill(double* arr, size_t len, double fillValue);
};

inline __host__ __device__ double NFmiGribPacking::ToPower(double value, double power)
{
	double divisor = 1.0;

	while (value < 0)
	{
		divisor /= power;
		value++;
	}

	while (value > 0)
	{
		divisor *= power;
		value--;
	}

	return divisor;
}

#endif /* HAVE_CUDA */
#endif /* NFMIGRIBPACKING_H */
