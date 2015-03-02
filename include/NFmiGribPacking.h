/* 
 * File:   NFmiGribPacking.h
 * Author: partio
 *
 * Created on February 20, 2015, 11:14 AM
 */

#ifndef NFMIGRIBPACKING_H
#define	NFMIGRIBPACKING_H

#ifdef HAVE_CUDA

#include <iostream>

#if defined __GNUC__ && __GNUC_MINOR__ > 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

#include <cuda_runtime.h>

#if defined __GNUC__ && __GNUC_MINOR__ > 5
#pragma GCC diagnostic pop
#endif

void CheckCudaError(cudaError_t errarg, const char* file, const int line);

#define CUDA_CHECK(errarg)	 CheckCudaError(errarg, __FILE__, __LINE__)

inline
void CheckCudaError(cudaError_t errarg, const char* file, const int line)
{
	if(errarg)
	{
		std::cerr << "Error at " << file << "(" << line << "): " << cudaGetErrorString(errarg) << std::endl;
		exit(1);
	}
}

#define BitMask1(i)	(1u << i)
#define BitTest(n,i)	!!((n) & BitMask1(i))

namespace NFmiGribPacking
{

struct packing_coefficients
{
	int bitsPerValue;
	double binaryScaleFactor;
	double decimalScaleFactor;
	double referenceValue;

	packing_coefficients()
		: bitsPerValue(0)
		, binaryScaleFactor(0)
		, decimalScaleFactor(0)
		, referenceValue(0)
	{}

};

namespace simple_packing
{
bool Unpack(double* d_arr, const unsigned char* d_packed, const int* d_bitmap, size_t unpackedLen, packing_coefficients coeffs, cudaStream_t& stream);
bool Pack(const double* d_arr, unsigned char* d_packed, const int* d_bitmap, size_t unpackedLength, packing_coefficients coeffs, cudaStream_t& stream);
long get_decimal_scale_fact(double max, double min, long bpval,long binary_scale);
long get_binary_scale_fact(double max, double min, long bpval);
}

namespace jpeg_packing
{
bool Unpack(double* d_arr, const unsigned char* d_packed, const int* d_bitmap, size_t unpackedLen, packing_coefficients coeffs, cudaStream_t& stream);
bool Pack();
}

void UnpackBitmap(const unsigned char* __restrict__ bitmap, int* __restrict__ unpacked, size_t len, size_t unpackedLen);

__host__ __device__
double ToPower(double value, double power);

};

inline __device__
double NFmiGribPacking::ToPower(double value, double power)
{
  double divisor = 1.0;

  while(value < 0)
  {
	divisor /= power;
	value++;
  }

  while(value > 0)
  {
	divisor *= power;
	value--;
  }

  return divisor;
}

#endif  /* HAVE_CUDA */
#endif	/* NFMIGRIBPACKING_H */

