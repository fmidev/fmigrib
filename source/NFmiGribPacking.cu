#include "NFmiGribPacking.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cuda_runtime_api.h>

void NFmiGribPacking::Fill(double* arr, size_t len, double fillValue)
{
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(arr);
	thrust::fill(ptr, ptr + len, fillValue);
}

template <typename T>
bool NFmiGribPacking::IsHostPointer(const T* ptr)
{
	cudaPointerAttributes attributes;
	cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

	bool ret;

	if (err == cudaErrorInvalidValue && ptr)
	{
		ret = true;

		// Clear error buffer
		cudaGetLastError();
	}
	else if (err == cudaSuccess)
	{
#if CUDART_VERSION >= 10010
		if (attributes.type == cudaMemoryTypeHost)
#else
		if (attributes.memoryType == cudaMemoryTypeHost)
#endif
		{
			ret = true;
		}
		else
		{
			ret = false;
		}
	}
	else
	{
		std::cerr << "simple_packing::Pack Error " << static_cast<int>(err) << " (" << cudaGetErrorString(err)
		          << ") while checking pointer attributes" << std::endl;
		exit(1);
	}

	return ret;
}

template bool NFmiGribPacking::IsHostPointer(const double*);
template bool NFmiGribPacking::IsHostPointer(const float*);

template <typename T>
__host__ __device__ void MinMax_(T* d, size_t unpackedLen, T& min, T& max)
{
	using namespace NFmiGribPacking::simple_packing;
	min = MissingValue<T>();
	max = MissingValue<T>();

	for (size_t i = 0; i < unpackedLen; i++)
	{
		T val = d[i];
		if (IsMissing(val))
			continue;

		if (val < min)
			min = val;
		if (val > max)
			max = val;
	}
}

template __host__ __device__ void MinMax_(double*, size_t, double&, double&);
template __host__ __device__ void MinMax_(float*, size_t, float&, float&);

template <typename T>
__global__ void MinMaxKernel(T* d, size_t unpackedLen, T& min, T& max)
{
	MinMax_<T>(d, unpackedLen, min, max);
}

template <typename T>
void NFmiGribPacking::MinMax(T* d, size_t unpackedLen, T& min, T& max, cudaStream_t& stream)
{
	if (IsHostPointer<T>(d))
	{
		MinMax_(d, unpackedLen, min, max);
	}
	else
	{
		double* d_min = 0;
		double* d_max = 0;
		CUDA_CHECK(cudaMalloc(&d_min, sizeof(double)));
		CUDA_CHECK(cudaMalloc(&d_max, sizeof(double)));

		MinMaxKernel<<<1, 1, 0, stream>>>(d, unpackedLen, min, max);

		CUDA_CHECK(cudaMemcpyAsync(&min, d_min, sizeof(double), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(&max, d_max, sizeof(double), cudaMemcpyDeviceToHost, stream));

		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUDA_CHECK(cudaFree(d_min));
		CUDA_CHECK(cudaFree(d_max));
	}
}

template void NFmiGribPacking::MinMax(double*, size_t, double&, double&, cudaStream_t&);
template void NFmiGribPacking::MinMax(float*, size_t, float&, float&, cudaStream_t&);
