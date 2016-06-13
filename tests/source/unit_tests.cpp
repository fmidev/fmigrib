#define BOOST_TEST_MAIN
#define  BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE NFmiGrib

#include "NFmiGrib.h"
#include <iostream>

const std::string grib1 = "file.grib";
const std::string grib2 = "file.grib2";
const std::string ens = "ens.grib2";

NFmiGrib reader;

void init1()
{
	bool open = reader.Open(grib1);

	BOOST_REQUIRE(open);

	bool next = reader.NextMessage();

	BOOST_REQUIRE(next);
}

void init2()
{
	bool open = reader.Open(grib2);

	BOOST_REQUIRE(open);

	bool next = reader.NextMessage();

	BOOST_REQUIRE(next);
}

void initens()
{
	bool open = reader.Open(ens);

	BOOST_REQUIRE(open);

	bool next = reader.NextMessage();

	BOOST_REQUIRE(next);
}

BOOST_AUTO_TEST_CASE(localDefinitionNumber)
{
	init1();

	BOOST_REQUIRE(reader.Message().LocalDefinitionNumber() == INVALID_INT_VALUE);

	init2();

	BOOST_REQUIRE(reader.Message().LocalDefinitionNumber() == 0);
}

BOOST_AUTO_TEST_CASE(productDefinitionTemplateNumber)
{
	init1();

	BOOST_REQUIRE(reader.Message().ProductDefinitionTemplateNumber() == INVALID_INT_VALUE);

	init2();

	reader.Message().ProductDefinitionTemplateNumber(1);
	BOOST_REQUIRE(reader.Message().ProductDefinitionTemplateNumber() == 1);

}

BOOST_AUTO_TEST_CASE(typeOfStatisticalProcessing)
{
	init1();

	BOOST_REQUIRE(reader.Message().TypeOfStatisticalProcessing() == INVALID_INT_VALUE);

	init2();

	reader.Message().ProductDefinitionTemplateNumber(8);
	reader.Message().TypeOfStatisticalProcessing(5);
	BOOST_REQUIRE(reader.Message().TypeOfStatisticalProcessing() == 5);

}

BOOST_AUTO_TEST_CASE(valuesLength)
{
	init1();

	BOOST_REQUIRE(reader.Message().ValuesLength() == 840480);

	init2();

	BOOST_REQUIRE(reader.Message().ValuesLength() == 243869);

}

BOOST_AUTO_TEST_CASE(dataDate)
{
	init1();

	BOOST_REQUIRE(reader.Message().DataDate() == 20120529);
	reader.Message().DataDate(20140912);
	BOOST_REQUIRE(reader.Message().DataDate() == 20140912);

	init2();

	BOOST_REQUIRE(reader.Message().DataDate() == 20120611);

}

BOOST_AUTO_TEST_CASE(dataTime)
{
	init1();

	BOOST_REQUIRE(reader.Message().DataTime() == 0);
	reader.Message().DataTime(600);
	BOOST_REQUIRE(reader.Message().DataTime() == 600);

	init2();

	BOOST_REQUIRE(reader.Message().DataTime() == 0);

}

BOOST_AUTO_TEST_CASE(forecastTime)
{
	init1();

	BOOST_REQUIRE(reader.Message().ForecastTime() == INVALID_INT_VALUE);

	init2();

	BOOST_REQUIRE(reader.Message().ForecastTime() == 9);
	reader.Message().ForecastTime(12);
	BOOST_REQUIRE(reader.Message().ForecastTime() == 12);
}

BOOST_AUTO_TEST_CASE(parameterUnit)
{
	init1();

	BOOST_REQUIRE(reader.Message().ParameterUnit() == "K");

	init2();

	BOOST_REQUIRE(reader.Message().ParameterUnit() == "K");
}

BOOST_AUTO_TEST_CASE(parameterNumber)
{
	init1();

	BOOST_REQUIRE(reader.Message().ParameterNumber() == 11);
	reader.Message().ParameterNumber(3);
	BOOST_REQUIRE(reader.Message().ParameterNumber() == 3);

	init2();

	BOOST_REQUIRE(reader.Message().ParameterNumber() == 0);
}

BOOST_AUTO_TEST_CASE(parameterCategory)
{
	init1();

	BOOST_REQUIRE(reader.Message().ParameterCategory() == INVALID_INT_VALUE);

	init2();

	BOOST_REQUIRE(reader.Message().ParameterCategory() == 0);
	reader.Message().ParameterCategory(3);
	BOOST_REQUIRE(reader.Message().ParameterCategory() == 3);

}

BOOST_AUTO_TEST_CASE(parameterDiscipline)
{
	init1();

	BOOST_REQUIRE(reader.Message().ParameterDiscipline() == INVALID_INT_VALUE);

	init2();

	BOOST_REQUIRE(reader.Message().ParameterDiscipline() == 0);
	reader.Message().ParameterDiscipline(10);
	BOOST_REQUIRE(reader.Message().ParameterDiscipline() == 10);

}

BOOST_AUTO_TEST_CASE(parameterName)
{
	init1();

	BOOST_REQUIRE(reader.Message().ParameterName() == "Temperature");

	init2();

	BOOST_REQUIRE(reader.Message().ParameterName() == "Temperature");

}

BOOST_AUTO_TEST_CASE(gridType)
{
	init1();

	BOOST_REQUIRE(reader.Message().GridType() == 10);
	reader.Message().GridType(1);
	BOOST_REQUIRE(reader.Message().GridType() == 1);

	init2();

	BOOST_REQUIRE(reader.Message().GridType() == 1);

}

BOOST_AUTO_TEST_CASE(gridOrientation)
{
	init1();

	BOOST_REQUIRE(reader.Message().GridOrientation() == INVALID_INT_VALUE);

	init2();

	BOOST_REQUIRE(reader.Message().GridOrientation() == INVALID_INT_VALUE);

}

BOOST_AUTO_TEST_CASE(iDirectionIncrement)
{
	init1();

	BOOST_REQUIRE(reader.Message().iDirectionIncrement() == 0.068);
	reader.Message().iDirectionIncrement(2);
	BOOST_REQUIRE(reader.Message().iDirectionIncrement() == 2);

	init2();

	BOOST_REQUIRE(reader.Message().iDirectionIncrement() == 0.125);

}

BOOST_AUTO_TEST_CASE(jDirectionIncrement)
{
	init1();

	BOOST_REQUIRE(reader.Message().jDirectionIncrement() == 0.068);
	reader.Message().jDirectionIncrement(2);
	BOOST_REQUIRE(reader.Message().jDirectionIncrement() == 2);

	init2();

	BOOST_REQUIRE(reader.Message().jDirectionIncrement() == 0.125);

}

BOOST_AUTO_TEST_CASE(levelType)
{
	init1();

	BOOST_REQUIRE(reader.Message().LevelType() == 109);
	reader.Message().LevelType(100);
	BOOST_REQUIRE(reader.Message().LevelType() == 100);

	init2();

	BOOST_REQUIRE(reader.Message().LevelType() == 105);

}

BOOST_AUTO_TEST_CASE(levelValue)
{
	init1();

	BOOST_REQUIRE(reader.Message().LevelValue() == 11);
	reader.Message().LevelValue(1);
	BOOST_REQUIRE(reader.Message().LevelValue() == 1);

	init2();

	BOOST_REQUIRE(reader.Message().LevelValue() == 40);

}

BOOST_AUTO_TEST_CASE(X0)
{
	init1();

	BOOST_REQUIRE(reader.Message().X0() == -33.5);
	reader.Message().X0(10.0);
	BOOST_REQUIRE(reader.Message().X0() == 10.0);

	init2();

	BOOST_REQUIRE(reader.Message().X0() == 334);

}

BOOST_AUTO_TEST_CASE(Y0)
{
	init1();

	BOOST_REQUIRE(reader.Message().Y0() == -24);
	reader.Message().Y0(10.0);
	BOOST_REQUIRE(reader.Message().Y0() == 10.0);

	init2();

	BOOST_REQUIRE(reader.Message().Y0() == 22.5);

}

BOOST_AUTO_TEST_CASE(X1)
{
	init1();

	BOOST_REQUIRE(reader.Message().X1() == 36.472);
	reader.Message().X1(10.0);
	BOOST_REQUIRE(reader.Message().X1() == 10.0);

	init2();

	BOOST_REQUIRE(reader.Message().X1() == 40);

}

BOOST_AUTO_TEST_CASE(Y1)
{
	init1();

	BOOST_REQUIRE(reader.Message().Y1() == 31.42);
	reader.Message().Y1(10.0);
	BOOST_REQUIRE(reader.Message().Y1() == 10.0);

	init2();

	BOOST_REQUIRE(reader.Message().Y1() == -35);

}

BOOST_AUTO_TEST_CASE(southPoleX)
{
	init1();

	BOOST_REQUIRE(reader.Message().SouthPoleX() == 0);
	reader.Message().SouthPoleX(5.0);
	BOOST_REQUIRE(reader.Message().SouthPoleX() == 5.0);

	init2();

	BOOST_REQUIRE(reader.Message().SouthPoleX() == 0);

}

BOOST_AUTO_TEST_CASE(southPoleY)
{
	init1();

	BOOST_REQUIRE(reader.Message().SouthPoleY() == -30);
	reader.Message().SouthPoleY(-40.0);
	BOOST_REQUIRE(reader.Message().SouthPoleY() == -40.0);

	init2();

	BOOST_REQUIRE(reader.Message().SouthPoleY() == -30);

}

BOOST_AUTO_TEST_CASE(edition)
{
	init1();

	BOOST_REQUIRE(reader.Message().Edition() == 1);
	reader.Message().Edition(2);
	BOOST_REQUIRE(reader.Message().Edition() == 2);

	init2();

	BOOST_REQUIRE(reader.Message().Edition() == 2);

}

BOOST_AUTO_TEST_CASE(process)
{
	init1();

	BOOST_REQUIRE(reader.Message().Process() == 1);
	reader.Message().Process(2);
	BOOST_REQUIRE(reader.Message().Process() == 2);

	init2();

	BOOST_REQUIRE(reader.Message().Process() == 141);

}

BOOST_AUTO_TEST_CASE(centre)
{
	init1();

	BOOST_REQUIRE(reader.Message().Centre() == 86);
	reader.Message().Centre(98);
	BOOST_REQUIRE(reader.Message().Centre() == 98);

	init2();

	BOOST_REQUIRE(reader.Message().Centre() == 98);

}

BOOST_AUTO_TEST_CASE(iScansNegatively)
{
	init1();

	BOOST_REQUIRE(!reader.Message().IScansNegatively());
	reader.Message().IScansNegatively(true);
	BOOST_REQUIRE(reader.Message().IScansNegatively());

	init2();

	BOOST_REQUIRE(!reader.Message().IScansNegatively());

}

BOOST_AUTO_TEST_CASE(jScansPositively)
{
	init1();

	BOOST_REQUIRE(reader.Message().JScansPositively());
	reader.Message().JScansPositively(false);
	BOOST_REQUIRE(!reader.Message().JScansPositively());

	init2();

	BOOST_REQUIRE(!reader.Message().JScansPositively());

}

BOOST_AUTO_TEST_CASE(sizeX)
{
	init1();

	BOOST_REQUIRE(reader.Message().SizeX() == 1030);
	reader.Message().SizeX(100);
	BOOST_REQUIRE(reader.Message().SizeX() == 100);

	init2();

	BOOST_REQUIRE(reader.Message().SizeX() == 529);

}

BOOST_AUTO_TEST_CASE(sizeY)
{
	init1();

	BOOST_REQUIRE(reader.Message().SizeY() == 816);
	reader.Message().SizeY(10);
	BOOST_REQUIRE(reader.Message().SizeY() == 10);

	init2();

	BOOST_REQUIRE(reader.Message().SizeY() == 461);

}

BOOST_AUTO_TEST_CASE(table2Version)
{
	init1();

	BOOST_REQUIRE(reader.Message().Table2Version() == 1);
	reader.Message().Table2Version(2);
	BOOST_REQUIRE(reader.Message().Table2Version() == 2);

	init2();

	BOOST_REQUIRE(reader.Message().Table2Version() == INVALID_INT_VALUE);

}

BOOST_AUTO_TEST_CASE(numberOfMissing)
{
	init1();

	BOOST_REQUIRE(reader.Message().NumberOfMissing() == 0);

	init2();

	BOOST_REQUIRE(reader.Message().NumberOfMissing() == 0);

}

BOOST_AUTO_TEST_CASE(normalizedLevelType)
{
	init1();

	BOOST_REQUIRE(reader.Message().NormalizedLevelType(1) == 109);
	BOOST_REQUIRE(reader.Message().NormalizedLevelType(2) == 105);

	init2();

	BOOST_REQUIRE(reader.Message().NormalizedLevelType(1) == 109);
	BOOST_REQUIRE(reader.Message().NormalizedLevelType(2) == 105);

}

BOOST_AUTO_TEST_CASE(normalizedGridType)
{
	init1();

	BOOST_REQUIRE(reader.Message().NormalizedGridType(1) == 10);
	BOOST_REQUIRE(reader.Message().NormalizedGridType(2) == 1);

	init2();

	BOOST_REQUIRE(reader.Message().NormalizedGridType(1) == 10);
	BOOST_REQUIRE(reader.Message().NormalizedGridType(2) == 1);

}

BOOST_AUTO_TEST_CASE(normalizedStep)
{
	init1();

	BOOST_REQUIRE(reader.Message().NormalizedStep(false, false) == 7);
	reader.Message().TimeRangeIndicator(5);
	BOOST_REQUIRE(reader.Message().NormalizedStep(false, true) == 7);
	BOOST_REQUIRE(reader.Message().NormalizedStep(true, true) == 0);
	reader.Message().UnitOfTimeRange(11);
	BOOST_REQUIRE(reader.Message().NormalizedStep(false, true) == 7*6);
	reader.Message().TimeRangeIndicator(10);
	reader.Message().UnitOfTimeRange(1);
	BOOST_REQUIRE(reader.Message().NormalizedStep(true, true) == 1792);

	init2();

	BOOST_REQUIRE(reader.Message().NormalizedStep(false, false) == 9);

}

BOOST_AUTO_TEST_CASE(normalizedUnitOfTimeRange)
{
	init1();

	BOOST_REQUIRE(reader.Message().NormalizedUnitOfTimeRange() == 1);

	init2();

	BOOST_REQUIRE(reader.Message().NormalizedUnitOfTimeRange() == 1);

}

BOOST_AUTO_TEST_CASE(startStep)
{
	init1();

	BOOST_REQUIRE(reader.Message().StartStep() == 7);
	reader.Message().StartStep(4);
	BOOST_REQUIRE(reader.Message().StartStep() == 4);

	init2();

	BOOST_REQUIRE(reader.Message().StartStep() == 9);

}

BOOST_AUTO_TEST_CASE(endStep)
{
	init1();

	BOOST_REQUIRE(reader.Message().EndStep() == 7);
	reader.Message().EndStep(8);
	BOOST_REQUIRE(reader.Message().EndStep() == 8);

	init2();

	BOOST_REQUIRE(reader.Message().EndStep() == 9);

}

BOOST_AUTO_TEST_CASE(stepUnits)
{
	init1();

	BOOST_REQUIRE(reader.Message().StepUnits() == 1);
	reader.Message().StepUnits(8);
	BOOST_REQUIRE(reader.Message().StepUnits() == 8);

	init2();

	BOOST_REQUIRE(reader.Message().StepUnits() == 1);

}

BOOST_AUTO_TEST_CASE(stepRange)
{
	init1();

	BOOST_REQUIRE(reader.Message().StepRange() == 7);
	reader.Message().StepRange(17);
	BOOST_REQUIRE(reader.Message().StepRange() == 17);

	init2();

	BOOST_REQUIRE(reader.Message().StepRange() == 9);

}

BOOST_AUTO_TEST_CASE(year)
{
	init1();

	BOOST_REQUIRE(reader.Message().Year() == 2012);

	init2();

	BOOST_REQUIRE(reader.Message().Year() == 2012);
	reader.Message().Year("2014");
	BOOST_REQUIRE(reader.Message().Year() == 2014);

}

BOOST_AUTO_TEST_CASE(month)
{
	init1();

	BOOST_REQUIRE(reader.Message().Month() == 5);
	reader.Message().Month("7");
	BOOST_REQUIRE(reader.Message().Month() == 7);

	init2();

	BOOST_REQUIRE(reader.Message().Month() == 6);

}

BOOST_AUTO_TEST_CASE(day)
{
	init1();

	BOOST_REQUIRE(reader.Message().Day() == 29);
	reader.Message().Day("7");
	BOOST_REQUIRE(reader.Message().Day() == 7);

	init2();

	BOOST_REQUIRE(reader.Message().Day() == 11);

}

BOOST_AUTO_TEST_CASE(hour)
{
	init1();

	BOOST_REQUIRE(reader.Message().Hour() == 0);
	reader.Message().Hour("7");
	BOOST_REQUIRE(reader.Message().Hour() == 7);

	init2();

	BOOST_REQUIRE(reader.Message().Hour() == 0);

}

BOOST_AUTO_TEST_CASE(minute)
{
	init1();

	BOOST_REQUIRE(reader.Message().Minute() == 0);
	reader.Message().Minute("17");
	BOOST_REQUIRE(reader.Message().Minute() == 17);

	init2();

	BOOST_REQUIRE(reader.Message().Minute() == 0);

}

BOOST_AUTO_TEST_CASE(second)
{
	init1();

	BOOST_REQUIRE(reader.Message().Second() == 0);
	reader.Message().Second("1");
	BOOST_REQUIRE(reader.Message().Second() == 1);

	init2();

	BOOST_REQUIRE(reader.Message().Second() == 0);

}

BOOST_AUTO_TEST_CASE(bitmap)
{
	init1();

	BOOST_REQUIRE(!reader.Message().Bitmap());
	reader.Message().Bitmap(true);
	BOOST_REQUIRE(reader.Message().Bitmap());

	init2();

	BOOST_REQUIRE(!reader.Message().Bitmap());

}

BOOST_AUTO_TEST_CASE(bitsPerValue)
{
	init1();

	BOOST_REQUIRE(reader.Message().BitsPerValue() == 10);
	reader.Message().BitsPerValue(16);
	BOOST_REQUIRE(reader.Message().BitsPerValue() == 16);

	init2();

	BOOST_REQUIRE(reader.Message().BitsPerValue() == 12);

}

BOOST_AUTO_TEST_CASE(uvRelativeToGrid)
{
	init1();

	BOOST_REQUIRE(reader.Message().UVRelativeToGrid());
	reader.Message().UVRelativeToGrid(false);
	BOOST_REQUIRE(!reader.Message().UVRelativeToGrid());

	init2();

	BOOST_REQUIRE(!reader.Message().UVRelativeToGrid());
	reader.Message().UVRelativeToGrid(true);
	BOOST_REQUIRE(reader.Message().UVRelativeToGrid());
	reader.Message().UVRelativeToGrid(false);
	BOOST_REQUIRE(!reader.Message().UVRelativeToGrid());

}

BOOST_AUTO_TEST_CASE(resolutionAndComponentFlags)
{
	init1();

	BOOST_REQUIRE(reader.Message().ResolutionAndComponentFlags() == 136);
	reader.Message().ResolutionAndComponentFlags(16);
	BOOST_REQUIRE(reader.Message().ResolutionAndComponentFlags() == 16);

	init2();

	BOOST_REQUIRE(reader.Message().ResolutionAndComponentFlags() == 48);

}

BOOST_AUTO_TEST_CASE(packingType)
{
	init1();

	BOOST_REQUIRE(reader.Message().PackingType() == "grid_simple");
	reader.Message().PackingType("grid_second_order");
	BOOST_REQUIRE(reader.Message().PackingType() == "grid_second_order");

	init2();

	BOOST_REQUIRE(reader.Message().PackingType() == "grid_simple");

}

BOOST_AUTO_TEST_CASE(typeOfGeneratingProcess)
{
	init1();

	BOOST_REQUIRE(reader.Message().TypeOfGeneratingProcess() == INVALID_INT_VALUE);
	
	init2();

	BOOST_REQUIRE(reader.Message().TypeOfGeneratingProcess() == 2);
	reader.Message().TypeOfGeneratingProcess(1);
	BOOST_REQUIRE(reader.Message().TypeOfGeneratingProcess() == 1);

}

BOOST_AUTO_TEST_CASE(xLengthInMeters)
{
	init1();

	BOOST_REQUIRE(reader.Message().XLengthInMeters() == INVALID_INT_VALUE);

	init2();

	BOOST_REQUIRE(reader.Message().XLengthInMeters() == INVALID_INT_VALUE);

}

BOOST_AUTO_TEST_CASE(yLengthInMeters)
{
	init1();

	BOOST_REQUIRE(reader.Message().YLengthInMeters() == INVALID_INT_VALUE);

	init2();

	BOOST_REQUIRE(reader.Message().YLengthInMeters() == INVALID_INT_VALUE);

}

BOOST_AUTO_TEST_CASE(unitOfTimeRange)
{
	init1();

	BOOST_REQUIRE(reader.Message().UnitOfTimeRange() == 1);
	reader.Message().UnitOfTimeRange(2);
	BOOST_REQUIRE(reader.Message().UnitOfTimeRange() == 2);

	init2();

	BOOST_REQUIRE(reader.Message().UnitOfTimeRange() == 1);

}

BOOST_AUTO_TEST_CASE(lengthOfTimeRange)
{
	init1();

	BOOST_REQUIRE(reader.Message().LengthOfTimeRange() == INVALID_INT_VALUE);

	init2();

	BOOST_REQUIRE(reader.Message().LengthOfTimeRange() == INVALID_INT_VALUE);

}

BOOST_AUTO_TEST_CASE(timeRangeIndicator)
{
	init1();

	BOOST_REQUIRE(reader.Message().TimeRangeIndicator() == 0);
	reader.Message().TimeRangeIndicator(10);
	BOOST_REQUIRE(reader.Message().TimeRangeIndicator() == 10);

	init2();

	BOOST_REQUIRE(reader.Message().TimeRangeIndicator() == 0);

}

BOOST_AUTO_TEST_CASE(P1)
{
	init1();

	BOOST_REQUIRE(reader.Message().P1() == 7);
	reader.Message().P1(10);
	BOOST_REQUIRE(reader.Message().P1() == 10);

	init2();

	BOOST_REQUIRE(reader.Message().P1() == INVALID_INT_VALUE);

}

BOOST_AUTO_TEST_CASE(P2)
{
	init1();

	BOOST_REQUIRE(reader.Message().P2() == 0);
	reader.Message().P2(10);
	BOOST_REQUIRE(reader.Message().P2() == 10);

	init2();

	BOOST_REQUIRE(reader.Message().P2() == INVALID_INT_VALUE);

}

BOOST_AUTO_TEST_CASE(NV)
{
	init1();

	BOOST_REQUIRE(reader.Message().NV() == 2);
	reader.Message().NV(4);
	BOOST_REQUIRE(reader.Message().NV() == 4);

	init2();

	BOOST_REQUIRE(reader.Message().NV() == 184);

}

BOOST_AUTO_TEST_CASE(PV)
{
	init1();

	BOOST_REQUIRE(reader.Message().PV(2, 0)[0] == 1.5831093750e+04);
	reader.Message().PV({0.1, 0.2}, 2);
	BOOST_CHECK_CLOSE(reader.Message().PV(2, 0)[0], 0.1, 0.01);
	BOOST_CHECK_CLOSE(reader.Message().PV(2, 0)[1], 0.2, 0.01);

	init2();

	BOOST_CHECK_CLOSE(reader.Message().PV(184, 66)[0], 16965.08984375, 0.01);

}

BOOST_AUTO_TEST_CASE(packedValuesLength)
{
	init1();

	BOOST_REQUIRE(reader.Message().PackedValuesLength() == 1050601);

	init2();

	BOOST_REQUIRE(reader.Message().PackedValuesLength() == 365804);

}

BOOST_AUTO_TEST_CASE(packedValues)
{
	init1();

	size_t N = reader.Message().PackedValuesLength();

	unsigned char* arr = new unsigned char[N];

	reader.Message().PackedValues(arr);

	BOOST_REQUIRE(arr[0] == 47);

	delete [] arr;

	init2();

	N = reader.Message().PackedValuesLength();
	arr = new unsigned char[N];	

	reader.Message().PackedValues(arr);

	BOOST_REQUIRE(arr[1] == 199);

	delete [] arr;

}

BOOST_AUTO_TEST_CASE(binaryScaleFactor)
{
	init1();

	BOOST_REQUIRE(reader.Message().BinaryScaleFactor() == -5);

	init2();

	BOOST_REQUIRE(reader.Message().BinaryScaleFactor() == -6);

}

BOOST_AUTO_TEST_CASE(decimalScaleFactor)
{
	init1();

	BOOST_REQUIRE(reader.Message().DecimalScaleFactor() == 0);

	init2();

	BOOST_REQUIRE(reader.Message().DecimalScaleFactor() == 0);

}

BOOST_AUTO_TEST_CASE(section4Length)
{
	init1();

	BOOST_REQUIRE(reader.Message().Section4Length() == 1050612);

	init2();

	BOOST_REQUIRE(reader.Message().Section4Length() == 770);

}

BOOST_AUTO_TEST_CASE(keyExists)
{
	init1();

	BOOST_REQUIRE(!reader.Message().KeyExists("asdfeaf"));
	BOOST_REQUIRE(reader.Message().KeyExists("gridType"));

	init2();

	BOOST_REQUIRE(reader.Message().KeyExists("edition"));

}

BOOST_AUTO_TEST_CASE(forecastType)
{
	init1();

	BOOST_REQUIRE(reader.Message().PerturbationNumber() == -999);
	BOOST_REQUIRE(reader.Message().ForecastType() == 1);
	BOOST_REQUIRE(reader.Message().ForecastTypeValue() == -999);

	reader.Message().ForecastType(2);
	BOOST_REQUIRE(reader.Message().ForecastType() == 1); // "analysis" is not set to grib1!

	init2();

	reader.Message().ForecastType(2);
	BOOST_REQUIRE(reader.Message().ForecastType() == 1); // "analysis" is set to grib2!

	reader.Message().ForecastType(4);
	BOOST_REQUIRE(reader.Message().ForecastType() == 4);

	reader.Message().ForecastTypeValue(50);
	BOOST_REQUIRE(reader.Message().ForecastTypeValue() == 50);

	initens();

	BOOST_REQUIRE(reader.Message().ForecastType() == 4);
	BOOST_REQUIRE(reader.Message().PerturbationNumber() == 0);
	BOOST_REQUIRE(reader.Message().ForecastTypeValue() == 0);
}


