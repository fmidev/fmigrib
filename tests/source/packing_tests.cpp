#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE grib_packing

#include <iostream>

#ifdef HAVE_CUDA
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"

#include <cuda_runtime.h>

#pragma GCC diagnostic pop

#include <cuda_runtime.h>

#include "NFmiGrib.h"
#include <iostream>

const std::string grib1 = "file.grib";
const std::string grib2 = "file.grib2";

const double kFloatMissing = 32700.;

NFmiGrib reader;

bool checkForDevice()
{
	int deviceCount;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);

	if (cudaResultCode == cudaErrorNoDevice || cudaResultCode != cudaSuccess) 
	{
        	return false;
	}

	return (deviceCount > 0);
}

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

BOOST_AUTO_TEST_CASE(simplePackGrib1)
{
	if (!checkForDevice()) return;

	init("file.grib");

	// get current data and then modify it

	double* arr = reader.Message().Values();
	size_t N = reader.Message().ValuesLength();

	BOOST_CHECK_CLOSE(arr[0], 216.478, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 221.301, 0.001);

	for (size_t i = 0; i < N; i++)
	{
		arr[i] += 100;
	}

	reader.Message().CudaPack(arr, N);
	reader.Message().Values(arr, N);

	delete [] arr;

	arr = reader.Message().Values();

	BOOST_CHECK_CLOSE(arr[0], 316.478, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 321.301, 0.001);
	delete [] arr;

}

BOOST_AUTO_TEST_CASE(simplePackGrib2)
{
	if (!checkForDevice()) return;

	init("file.grib2");

	// get current data and then modify it

	double* arr = reader.Message().Values();
	size_t N = reader.Message().ValuesLength();

	BOOST_CHECK_CLOSE(arr[0], 224.199, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 216.167, 0.001);

	for (size_t i = 0; i < N; i++)
	{
		arr[i] += 100;
	}

	reader.Message().CudaPack(arr, N);
	reader.Message().Values(arr, N);

	delete [] arr;

	arr = reader.Message().Values();

	BOOST_CHECK_CLOSE(arr[0], 324.199, 0.001);
	BOOST_CHECK_CLOSE(mean(arr, N), 316.167, 0.001);
	delete [] arr;

}

BOOST_AUTO_TEST_CASE(dummy)
{
	std::cerr << "cuda disabled\n";
}
#endif
