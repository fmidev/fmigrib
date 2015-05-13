#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE external_packing_tests

#include "NFmiGrib.h"
#include <iostream>

const std::string grib1 = "file.grib.gz";
const std::string grib2 = "file.grib2.bz2";

const double kFloatMissing = 32700.;

NFmiGrib reader;

void init(const std::string& fileName)
{
	BOOST_REQUIRE(reader.Open(fileName));
	BOOST_REQUIRE(reader.NextMessage());
}

BOOST_AUTO_TEST_CASE(readAndWriteGzip)
{
	init(grib1);

	double ref = reader.Message().ReferenceValue();

	reader.Message().Write("test.grib.gz");
	
	init("test.grib.gz");

	double newref = reader.Message().ReferenceValue();

	BOOST_REQUIRE(ref == newref);
}

BOOST_AUTO_TEST_CASE(readAndWriteBzip2)
{
	init(grib2);

	double ref = reader.Message().ReferenceValue();

	reader.Message().Write("test.grib2.bz2");
	
	init("test.grib2.bz2");

	double newref = reader.Message().ReferenceValue();

	BOOST_REQUIRE(ref == newref);
}
