#include "NFmiGribPacking.h"

bool NFmiGribPacking::IsHostPointer(const double* ptr)
{
	cudaPointerAttributes attributes;
	cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

	bool ret;

	if (err == cudaErrorInvalidValue && ptr)
	{
#ifdef DEBUG
		std::cerr << "CudaPack: Host memory was allocated with malloc" << std::endl;
#endif
		ret = true;

		// Clear error buffer
		cudaGetLastError();
	}
	else if (err == cudaSuccess)
	{
		if (attributes.memoryType == cudaMemoryTypeHost)
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
		std::cerr << "simple_packing::Pack Error " << static_cast<int> (err) << " (" << cudaGetErrorString(err) << ") while checking pointer attributes" << std::endl;
		exit(1);
	}

	return ret;
}

__host__ __device__
void MinMax_(double* d, size_t unpackedLen, double& min, double& max)
{
	min = 1e38;
	max = -1e38;
	const double kFloatMissing = 32700.;

	for (size_t i = 0; i < unpackedLen; i++)
	{
		double val = d[i];
		if (val == kFloatMissing) continue;

		if (val < min) min = val;
		if (val > max) max = val;
	}
}

__global__
void MinMaxKernel(double* d, size_t unpackedLen, double& min, double& max)
{
	MinMax_(d, unpackedLen, min, max);
}

void NFmiGribPacking::MinMax(double* d, size_t unpackedLen, double& min, double& max, cudaStream_t& stream)
{
	if (IsHostPointer(d))
	{
		MinMax_(d, unpackedLen, min, max);
	}
	else
	{
		double* d_min = 0;
		double* d_max = 0;
		CUDA_CHECK(cudaMalloc(&d_min, sizeof(double)));
		CUDA_CHECK(cudaMalloc(&d_max, sizeof(double)));
		
		MinMaxKernel<<<1, 1, 0, stream>>> (d, unpackedLen, min, max);

		CUDA_CHECK(cudaMemcpyAsync(&min, d_min, sizeof(double), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(&max, d_max, sizeof(double), cudaMemcpyDeviceToHost, stream));

		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUDA_CHECK(cudaFree(d_min));
		CUDA_CHECK(cudaFree(d_max));
	}
}
