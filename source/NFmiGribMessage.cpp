/*
 * NFmiGribMessageMessage.cpp
 *
 *  Created on: Oct 16, 2012
 *      Author: partio
 */

#include "NFmiGribMessage.h"
#include <stdexcept>

const long INVALID_INT_VALUE = -999;
const float kFloatMissing = 32700;

bool NFmiGribMessage::Read(grib_handle *h) {

  long t = 0;

  Clear();

  GRIB_CHECK(grib_get_long(h, "totalLength", &itsTotalLength), 0);

  GRIB_CHECK(grib_get_long(h,"editionNumber",&itsEdition),0);
  GRIB_CHECK(grib_get_long(h,"centre",&itsCentre),0);
  GRIB_CHECK(grib_get_long(h,"generatingProcessIdentifier",&itsProcess),0);

  GRIB_CHECK(grib_get_long(h,"dataDate",&itsDate),0);
  GRIB_CHECK(grib_get_long(h,"dataTime",&itsTime),0);

  GRIB_CHECK(grib_get_long(h,"Ni",&itsXSize),0);
  GRIB_CHECK(grib_get_long(h,"Nj",&itsYSize),0);

  GRIB_CHECK(grib_get_long(h,"iScansNegatively",&itsIScansNegatively),0);
  GRIB_CHECK(grib_get_long(h,"jScansPositively",&itsJScansPositively),0);

  GRIB_CHECK(grib_get_double(h,"latitudeOfFirstGridPointInDegrees",&itsLatitudeOfFirstGridPoint),0);
  GRIB_CHECK(grib_get_double(h,"longitudeOfFirstGridPointInDegrees",&itsLongitudeOfFirstGridPoint),0);

  // TODO: Copied from decode_grib_api, is this needed ?

  if (itsLongitudeOfFirstGridPoint == 180)
    itsLongitudeOfFirstGridPoint = -180;

  GRIB_CHECK(grib_get_long(h,"stepUnits",&itsStepUnits),0);
  GRIB_CHECK(grib_get_long(h,"stepRange",&itsStepRange),0);

  GRIB_CHECK(grib_get_long(h,"level",&itsLevel),0);

  GRIB_CHECK(grib_get_long(h,"bitmapPresent",&itsBitmapPresent),0);

  GRIB_CHECK(grib_get_long(h, "year", &itsYear), 0);
  GRIB_CHECK(grib_get_long(h, "month", &itsMonth), 0);
  GRIB_CHECK(grib_get_long(h, "day", &itsDay), 0);
  GRIB_CHECK(grib_get_long(h, "hour", &itsHour), 0);
  GRIB_CHECK(grib_get_long(h, "minute", &itsMinute), 0);

  GRIB_CHECK(grib_get_long(h,"timeRangeIndicator", &itsTimeRangeIndicator), 0);

  // Edition-specific keys

  if (itsEdition == 1) {

    GRIB_CHECK(grib_get_long(h,"indicatorOfParameter",&itsIndicatorOfParameter),0);
    GRIB_CHECK(grib_get_long(h,"indicatorOfTypeOfLevel",&itsIndicatorOfTypeOfLevel),0);

    GRIB_CHECK(grib_get_long(h,"startStep",&itsStartStep),0);
    GRIB_CHECK(grib_get_long(h,"endStep",&itsEndStep),0);

    GRIB_CHECK(grib_get_long(h,"dataRepresentationType",&itsGridType),0);

    GRIB_CHECK(grib_get_long(h,"table2Version", &itsTable2Version), 0);

    t = 0;

    if (grib_get_long(h, "localDefinitionNumber", &t) == GRIB_SUCCESS)
      itsLocalDefinitionNumber = t;

    t = 0;

    if (grib_get_long(h, "type", &t) == GRIB_SUCCESS)
      itsDataType = t;

  }
  else if (itsEdition == 2) {

    GRIB_CHECK(grib_get_long(h,"typeOfFirstFixedSurface",&itsTypeOfFirstFixedSurface),0);
    GRIB_CHECK(grib_get_long(h,"discipline",&itsParameterDiscipline),0);
    GRIB_CHECK(grib_get_long(h,"parameterCategory",&itsParameterCategory),0);
    GRIB_CHECK(grib_get_long(h,"parameterNumber",&itsParameterNumber),0);

    GRIB_CHECK(grib_get_long(h,"gridDefinitionTemplateNumber",&itsGridDefinitionTemplate),0);
    GRIB_CHECK(grib_get_long(h,"forecastTime",&itsForecastTime),0);

    GRIB_CHECK(grib_get_long(h,"productDefinitionTemplateNumber", &itsLocalDefinitionNumber), 0);

    GRIB_CHECK(grib_get_long(h,"startStep", &itsStartStep), 0);
    GRIB_CHECK(grib_get_long(h,"endStep", &itsEndStep), 0);

  }
  else
    throw std::runtime_error("Unknown grib edition");

  t = 0;

  if (grib_get_long(h, "perturbationNumber", &t) == GRIB_SUCCESS)
    itsPerturbationNumber = t;

  t = 0;

  if (grib_get_long(h, "typeOfEnsembleForecast", &t) == GRIB_SUCCESS)
    itsTypeOfEnsembleForecast = t;

  t = 0;

  if (grib_get_long(h, "derivedForecast", &t) == GRIB_SUCCESS)
    itsDerivedForecast = t;

  t = 0;

  if (grib_get_long(h, "numberOfForecastsInTheEnsemble", &t) == GRIB_SUCCESS)
    itsNumberOfForecastsInTheEnsemble = t;

  t = 0;

  if (grib_get_long(h, "clusterIdentifier", &t) == GRIB_SUCCESS)
    itsClusterIdentifier = t;

  t = 0;

  if (grib_get_long(h, "forecastProbabilityNumber", &t) == GRIB_SUCCESS)
    itsForecastProbabilityNumber = t;

  t = 0;

  if (grib_get_long(h, "probabilityType", &t) == GRIB_SUCCESS)
    itsProbabilityType = t;

  t = 0;

  if (grib_get_long(h, "percentileValue", &t) == GRIB_SUCCESS)
    itsPercentileValue = t;

  t = 0;

  if (grib_get_long(h, "numberOfTimeRange", &t) == GRIB_SUCCESS)
    itsNumberOfTimeRange = t;

  t = 0;

  if (grib_get_long(h, "typeOfTimeIncrement", &t) == GRIB_SUCCESS)
    itsTypeOfTimeIncrement = t;


  size_t len = 255;
  char name[1024];

  GRIB_CHECK(grib_get_string(h, "parameterName", name, &len), 0);

  itsParameterName = name;

  // Projection-specific keys

  int gridType = NormalizedGridType();

  if (gridType == 0 || gridType == 10) { // latlon or rot latlon
    GRIB_CHECK(grib_get_double(h,"latitudeOfLastGridPointInDegrees",&itsLatitudeOfLastGridPoint),0);
    GRIB_CHECK(grib_get_double(h,"longitudeOfLastGridPointInDegrees",&itsLongitudeOfLastGridPoint),0);

    GRIB_CHECK(grib_get_double(h,"iDirectionIncrementInDegrees",&itsXResolution),0);
    GRIB_CHECK(grib_get_double(h,"jDirectionIncrementInDegrees",&itsYResolution),0);

    if (gridType == 10) {
      GRIB_CHECK(grib_get_double(h,"latitudeOfSouthernPoleInDegrees",&itsLatitudeOfSouthernPole),0);
      GRIB_CHECK(grib_get_double(h,"longitudeOfSouthernPoleInDegrees",&itsLongitudeOfSouthernPole),0);
    }
  }
  else if (gridType == 5) { // (polar) Stereographic
    GRIB_CHECK(grib_get_double(h,"orientationOfTheGrid",&itsOrientationOfTheGrid),0);

    GRIB_CHECK(grib_get_double(h,"yDirectionGridLengthInMetres",&itsYResolution),0);
    GRIB_CHECK(grib_get_double(h,"xDirectionGridLengthInMetres",&itsXResolution),0);

  }
  else
    throw std::runtime_error("Unsupported projection");


  // Grib values

  GRIB_CHECK(grib_get_size(h,"values",&itsValuesLength),0);

  // Set missing value to kFloatMissing

  if (itsBitmapPresent == 1)
    GRIB_CHECK(grib_set_double(h,"missingValue",kFloatMissing),0);

  return true;
}

/*
 * Values()
 *
 * Returns a pointer to grib data values, caller is responsible
 * for freeing memory.
 *
 * TODO: Should this function return the data in a vector
 * (like NFmiNetCDF library does)
 */

double *NFmiGribMessage::Values() {
  if (!itsValues) {
    itsValues = static_cast<double*> (malloc(itsValuesLength*sizeof(double)));

    GRIB_CHECK(grib_get_double_array(h,"values",itsValues,&itsValuesLength),0);
  }

  return itsValues;
}

int NFmiGribMessage::ValuesLength() {
  return static_cast<int> (itsValuesLength);
}

long NFmiGribMessage::DataDate() {
  return itsDate;
}

long NFmiGribMessage::DataTime() {
  return itsTime;
}

long NFmiGribMessage::ForecastTime() {
  return itsForecastTime;
}

long NFmiGribMessage::ParameterNumber() {

  if (itsEdition == 1)
    return itsIndicatorOfParameter;
  else
    return itsParameterNumber;
}

long NFmiGribMessage::ParameterCategory() {
  if (itsEdition == 2)
    return itsParameterCategory;
  else
    return INVALID_INT_VALUE;
}

long NFmiGribMessage::ParameterDiscipline() {
  if (itsEdition == 2)
    return itsParameterDiscipline;
  else
    return INVALID_INT_VALUE;
}

std::string NFmiGribMessage::ParameterName() {
  return itsParameterName;
}

long NFmiGribMessage::GridType() {
  if (itsEdition == 1)
    return itsGridType;
  else
    return itsGridDefinitionTemplate;
}

double NFmiGribMessage::GridOrientation() {
  return itsOrientationOfTheGrid;
}

long NFmiGribMessage::LevelType() {
  if (itsEdition == 2)
    return itsTypeOfFirstFixedSurface;
  else
    return itsIndicatorOfTypeOfLevel;
}

double NFmiGribMessage::XResolution() {
  return itsXResolution;
}

double NFmiGribMessage::YResolution() {
  return itsYResolution;
}

/*
 * Clear()
 *
 * This function is every time a new grib message is read. It
 * initializes all grib message specific variables.
 */

void NFmiGribMessage::Clear() {

  itsValues = 0;
  itsValuesLength = 0;

  itsEdition = INVALID_INT_VALUE;
  itsProcess = INVALID_INT_VALUE;
  itsCentre = INVALID_INT_VALUE;

  itsXSize = INVALID_INT_VALUE;
  itsYSize = INVALID_INT_VALUE;

  itsLatitudeOfFirstGridPoint = INVALID_INT_VALUE;
  itsLongitudeOfFirstGridPoint = INVALID_INT_VALUE;
  itsLatitudeOfLastGridPoint = INVALID_INT_VALUE;
  itsLongitudeOfLastGridPoint = INVALID_INT_VALUE;

  itsIScansNegatively = INVALID_INT_VALUE;
  itsJScansPositively = INVALID_INT_VALUE;

  itsLatitudeOfSouthernPole = INVALID_INT_VALUE;
  itsLongitudeOfSouthernPole = INVALID_INT_VALUE;

  itsIndicatorOfParameter = INVALID_INT_VALUE;
  itsIndicatorOfTypeOfLevel = INVALID_INT_VALUE;

  itsLevel = INVALID_INT_VALUE;
  itsDate = INVALID_INT_VALUE;
  itsTime = INVALID_INT_VALUE;

  itsStepUnits = INVALID_INT_VALUE;
  itsStepRange = INVALID_INT_VALUE;
  itsStartStep = INVALID_INT_VALUE;
  itsEndStep = INVALID_INT_VALUE;

  itsGridType = INVALID_INT_VALUE;
  itsGridDefinitionTemplate = INVALID_INT_VALUE;

  itsValuesLength = INVALID_INT_VALUE;

  itsParameterDiscipline = INVALID_INT_VALUE;
  itsParameterCategory = INVALID_INT_VALUE;
  itsParameterNumber = INVALID_INT_VALUE;
  itsTypeOfFirstFixedSurface = INVALID_INT_VALUE;

  itsXResolution = INVALID_INT_VALUE;
  itsYResolution = INVALID_INT_VALUE;

  itsOrientationOfTheGrid = INVALID_INT_VALUE;
  itsBitmapPresent = INVALID_INT_VALUE;

  itsParameterName = "";
  itsForecastTime = INVALID_INT_VALUE;

  itsTotalLength = INVALID_INT_VALUE;
  itsTable2Version = INVALID_INT_VALUE;
  itsYear = INVALID_INT_VALUE;
  itsMonth = INVALID_INT_VALUE;
  itsDay = INVALID_INT_VALUE;
  itsHour = INVALID_INT_VALUE;
  itsMinute = INVALID_INT_VALUE;

  itsTimeRangeIndicator = INVALID_INT_VALUE;
  itsLocalDefinitionNumber = INVALID_INT_VALUE;

  itsDataType = INVALID_INT_VALUE;
  itsPerturbationNumber = INVALID_INT_VALUE;
  itsTypeOfEnsembleForecast = INVALID_INT_VALUE;
  itsDerivedForecast = INVALID_INT_VALUE;
  itsNumberOfForecastsInTheEnsemble = INVALID_INT_VALUE;
  itsClusterIdentifier = INVALID_INT_VALUE;
  itsForecastProbabilityNumber = INVALID_INT_VALUE;
  itsProbabilityType = INVALID_INT_VALUE;
  itsPercentileValue = INVALID_INT_VALUE;
  itsNumberOfTimeRange = INVALID_INT_VALUE;
  itsTypeOfTimeIncrement = INVALID_INT_VALUE;

}

double NFmiGribMessage::X0() {
  return itsLongitudeOfFirstGridPoint;
}

double NFmiGribMessage::Y0() {
  return itsLatitudeOfFirstGridPoint;
}

double NFmiGribMessage::SouthPoleX() {
  return itsLongitudeOfSouthernPole;
}

double NFmiGribMessage::SouthPoleY() {
  return itsLatitudeOfSouthernPole;
}

long NFmiGribMessage::Edition() {
  return itsEdition;
}

long NFmiGribMessage::SizeX() {
  return itsXSize;
}

long NFmiGribMessage::SizeY() {
  return itsYSize;
}

/*
 * NormalizedLevelType()
 *
 * Return level type number thats normalized to grib 1 definitions,
 * when that's possible.
 */

long NFmiGribMessage::NormalizedLevelType() {

  long type;

  if (itsEdition == 1)
    type = itsIndicatorOfTypeOfLevel;

  else {
    switch (itsTypeOfFirstFixedSurface) {
      case 100: // isobaric
      case 160: // depth below sea level
        type = itsTypeOfFirstFixedSurface;
        break;

      case 101: // mean sea
    	type = 102;
    	break;

      case 102: // specific altitude above mean-sea level
    	type = 103;
    	break;

      case 103: // specified height above ground
    	type = 105;
    	break;

      case 105: // hybrid
        type = 109;
        break;

      case 106: // depth below land surface
    	 type = 111;
    	 break;

      default:
         type = itsTypeOfFirstFixedSurface;
         break;

    }
  }

  return type;
}

/*
 * NormalizedGridType()
 *
 * Return geom type number thats normalized to grib 1 definitions,
 * when that's possible.
 */

long NFmiGribMessage::NormalizedGridType() {

  long type;

  if (itsEdition == 1)
    type = GridType();

  else {
    switch (GridType()) {
      case 0: // ll
      case 1: // rll
      case 2: // stretched ll
      case 3: // stretched rll
        type = 10 * GridType();
        break;

      case 20: // polar stereographic
        type = 5;
        break;

      case 30: // lambert conformal
      case 40: // gaussian ll
      	type = GridType() / 10;
        break;

      default:
         type = GridType();
         break;

    }
  }

  return type;
}
