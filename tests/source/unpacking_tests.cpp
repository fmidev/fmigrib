#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE grib_unpacking

#include <iostream>

#ifdef HAVE_CUDA
#ifdef __GNUC__
#if __GNUC_MINOR__ > 5
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

#include <cuda_runtime.h>

#if defined __GNUC__ && __GNUC_MINOR__ > 5
#pragma GCC diagnostic pop
#endif // __GNUC__

#include <cuda_runtime.h>

#include "NFmiGrib.h"
#include <iostream>

const std::string grib1 = "file.grib";
const std::string grib2 = "file.grib2";

const double kFloatMissing = 32700.;

NFmiGrib reader;

void init(const std::string& fileName)
{
	BOOST_REQUIRE(reader.Open(fileName));
	BOOST_REQUIRE(reader.NextMessage());
}

double mean(double* arr, size_t N)
{
	double mean = 0;
	size_t count = 0;

	for (size_t i = 0; i < N; i++)
	{
		double d = arr[i];
		if (d == kFloatMissing) continue;

		count++;
		mean += d;
	}

	if (count == 0) return 0;

	return mean / static_cast<double> (count);
}

BOOST_AUTO_TEST_CASE(simpleUnpackingWithMallocGrib1)
{

	init("file.grib");

	size_t N = reader.Message().ValuesLength();
	BOOST_REQUIRE(N == 840480);

	double* arr = reinterpret_cast<double*> (malloc(sizeof(double) * N));
	BOOST_REQUIRE(arr);

	reader.Message().CudaUnpack(arr, N);

	BOOST_CHECK_CLOSE(arr[0], 216.478, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 221.301, 0.001);

	free(arr);
}

BOOST_AUTO_TEST_CASE(simpleUnpackingGrib1)
{

	init("file.grib");

	// normal way

	double* arr = reader.Message().Values();
	size_t N = reader.Message().ValuesLength();

	BOOST_CHECK_CLOSE(arr[0], 216.478, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 221.301, 0.001);

	free (arr);

	// cuda way

	BOOST_REQUIRE(cudaSuccess == cudaMallocHost(&arr, N * sizeof(double)));

	reader.Message().CudaUnpack(arr, N);

	BOOST_CHECK_CLOSE(arr[0], 216.478, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 221.301, 0.001);

	BOOST_REQUIRE(cudaSuccess == cudaFreeHost(arr));
}

BOOST_AUTO_TEST_CASE(simpleUnpackingGrib2)
{

	init("file.grib2");

	// normal way

	double* arr = reader.Message().Values();
	size_t N = reader.Message().ValuesLength();

	BOOST_CHECK_CLOSE(arr[0], 224.199, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 216.167, 0.001);

	free (arr);

	// cuda way

	BOOST_REQUIRE(cudaSuccess == cudaMallocHost(&arr, N * sizeof(double)));

	reader.Message().CudaUnpack(arr, N);

	BOOST_CHECK_CLOSE(arr[0], 224.199, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 216.167, 0.001);

	BOOST_REQUIRE(cudaSuccess == cudaFreeHost(arr));
	

}

BOOST_AUTO_TEST_CASE(simpleUnpackingDevicePointerGrib1)
{
	init("file.grib");

	double* d_arr = 0;
	size_t N = reader.Message().ValuesLength();

	BOOST_REQUIRE(cudaSuccess == cudaMalloc(&d_arr, N * sizeof(double)));

	reader.Message().CudaUnpack(d_arr, N);

	double* arr = new double[N];

	BOOST_REQUIRE(cudaSuccess == cudaMemcpy(arr, d_arr, sizeof(double) * N, cudaMemcpyDeviceToHost));

	BOOST_CHECK_CLOSE(arr[0], 216.478, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 221.301, 0.001);

	BOOST_REQUIRE(cudaSuccess == cudaFree(d_arr));

	delete [] arr;
}

BOOST_AUTO_TEST_CASE(simpleUnpackingDevicePointerGrib2)
{

	init("file.grib2");

	double* d_arr = 0;
	size_t N = reader.Message().ValuesLength();
	
	BOOST_REQUIRE(cudaSuccess == cudaMalloc(&d_arr, N * sizeof(double)));

	reader.Message().CudaUnpack(d_arr, N);

	double* arr = new double[N];

	BOOST_REQUIRE(cudaSuccess == cudaMemcpy(arr, d_arr, sizeof(double) * N, cudaMemcpyDeviceToHost));

	BOOST_CHECK_CLOSE(arr[0], 224.199, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 216.167, 0.001);

	BOOST_REQUIRE(cudaSuccess == cudaFree(d_arr));

	delete [] arr;

}

BOOST_AUTO_TEST_CASE(simpleUnpackingWithStreamGrib1)
{

	init("file.grib");

	double* d_arr = 0;
	size_t N = reader.Message().ValuesLength();

	BOOST_REQUIRE(cudaSuccess == cudaMalloc(&d_arr, N * sizeof(double)));

	cudaStream_t stream;

	BOOST_REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

	reader.Message().CudaUnpack(d_arr, N, stream);

	double* arr = new double[N];

	BOOST_REQUIRE(cudaSuccess == cudaMemcpyAsync(arr, d_arr, sizeof(double) * N, cudaMemcpyDeviceToHost));

	BOOST_REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

	BOOST_CHECK_CLOSE(arr[0], 216.478, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 221.301, 0.001);

	BOOST_REQUIRE(cudaSuccess == cudaFree(d_arr));
	BOOST_REQUIRE(cudaSuccess == cudaStreamDestroy(stream));

	delete [] arr;
}

BOOST_AUTO_TEST_CASE(jpegUnpackingGrib2)
{

	init("file_jpeg.grib2");

	// normal way

	double* arr = reader.Message().Values();
	size_t N = reader.Message().ValuesLength();

	BOOST_CHECK_CLOSE(arr[0], 216.478, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 221.301, 0.001);

	// cuda way

	double* arr2;

	BOOST_REQUIRE(cudaSuccess == cudaMallocHost(&arr2, N * sizeof(double)));

	reader.Message().CudaUnpack(arr2, N);

	BOOST_CHECK_CLOSE(arr2[0], 216.478, 0.001);

	// comprimato watermark messes up with mean
	//BOOST_CHECK_CLOSE(mean(arr2, N), 216.167, 0.001);

	free (arr);
	BOOST_REQUIRE(cudaSuccess == cudaFreeHost(arr2));

}

#else
BOOST_AUTO_TEST_CASE(dummy)
{
	std::cerr << "cuda disabled\n";
}
#endif
