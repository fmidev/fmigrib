#include "NFmiGrib.h"
#include <stdexcept>
#include <iostream>

const int INVALID_INT_VALUE = -999;
const float kFloatMissing = 32700;

NFmiGrib::NFmiGrib() :
  h(0),
  f(0),
  itsMessageCount(0),
  itsCurrentMessage(0) {}

NFmiGrib::NFmiGrib(const std::string &theFileName) :
  h(0),
  f(0),
  itsMessageCount(0),
  itsCurrentMessage(0) {

  Open(theFileName);
}

NFmiGrib::~NFmiGrib() {

  if (h)
    grib_handle_delete(h);

  if (f)
    fclose(f);
}

bool NFmiGrib::Open(const std::string &theFileName) {

   // Open with 'rb', although in linux it equals to 'r'

  if (f)
    fclose(f);

  if (!(f = fopen(theFileName.c_str(), "rb")))
    return false;

  if (grib_count_in_file(0, f, &itsMessageCount) != GRIB_SUCCESS)
    return false;

  if (h)
    grib_handle_delete(h);

  grib_multi_support_on(0); // Multigrib support on

  return true;
}

long NFmiGrib::SizeX() {
  return itsXSize;
}

long NFmiGrib::SizeY() {
  return itsYSize;
}

bool NFmiGrib::Read() {

  if (!f)
    return false;

  if (!h)
    return false;

  Clear();

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

  // Edition-specific keys

  if (itsEdition == 1) {

    GRIB_CHECK(grib_get_long(h,"indicatorOfParameter",&itsIndicatorOfParameter),0);
    GRIB_CHECK(grib_get_long(h,"indicatorOfTypeOfLevel",&itsIndicatorOfTypeOfLevel),0);

    GRIB_CHECK(grib_get_long(h,"startStep",&itsStartStep),0);
    GRIB_CHECK(grib_get_long(h,"endStep",&itsEndStep),0);

    GRIB_CHECK(grib_get_long(h,"dataRepresentationType",&itsGridType),0);

  }
  else if (itsEdition == 2) {

    GRIB_CHECK(grib_get_long(h,"typeOfFirstFixedSurface",&itsTypeOfFirstFixedSurface),0);
    GRIB_CHECK(grib_get_long(h,"discipline",&itsParameterDiscipline),0);
    GRIB_CHECK(grib_get_long(h,"parameterCategory",&itsParameterCategory),0);
    GRIB_CHECK(grib_get_long(h,"parameterNumber",&itsParameterNumber),0);

    GRIB_CHECK(grib_get_long(h,"gridDefinitionTemplateNumber",&itsGridDefinitionTemplate),0);
    GRIB_CHECK(grib_get_long(h,"forecastTime",&itsForecastTime),0);
  }
  else 
    throw std::runtime_error("Unknown grib edition");

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

  return true;
}

bool NFmiGrib::NextMessage() {

  int err;

  if ((h = grib_handle_new_from_file(0,f,&err)) != NULL) {
    itsCurrentMessage++;

    return true;

  }
  else
    return false;
}

int NFmiGrib::MessageCount() {
  return itsMessageCount;
}

int NFmiGrib::CurrentMessageIndex() {
  return itsCurrentMessage;
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

double *NFmiGrib::Values() {

  // Set missing value to kFloatMissing

  if (itsBitmapPresent == 1)
    GRIB_CHECK(grib_set_double(h,"missingValue",kFloatMissing),0);

  GRIB_CHECK(grib_get_size(h,"values",&itsValuesLength),0);

  itsValues = (double*) malloc(itsValuesLength*sizeof(double));

  GRIB_CHECK(grib_get_double_array(h,"values",itsValues,&itsValuesLength),0);

  return itsValues;
}

int NFmiGrib::ValuesLength() {

  if ((int) itsValuesLength < 0)
    GRIB_CHECK(grib_get_size(h,"values",&itsValuesLength),0);

  return (int) itsValuesLength;
}

long NFmiGrib::DataDate() {
  return itsDate;
}

long NFmiGrib::DataTime() {
  return itsTime;
}

long NFmiGrib::ForecastTime() {
  return itsForecastTime;
}

long NFmiGrib::ParameterNumber() {

  if (itsEdition == 1)
    return itsIndicatorOfParameter;
  else
    return itsParameterNumber;
}

long NFmiGrib::ParameterCategory() {
  if (itsEdition == 2)
    return itsParameterCategory;
  else
    return INVALID_INT_VALUE;
}

long NFmiGrib::ParameterDiscipline() {
  if (itsEdition == 2)
    return itsParameterDiscipline;
  else
    return INVALID_INT_VALUE;
}

std::string NFmiGrib::ParameterName() {
  return itsParameterName;
}

long NFmiGrib::GridType() {
  if (itsEdition == 1)
    return itsGridType;
  else
    return itsGridDefinitionTemplate;
}

double NFmiGrib::GridOrientation() {
  return itsOrientationOfTheGrid;
}

/*
 * NormalizedGridType()
 *
 * Return grid type number thats normalized to grib 1 definitions,
 * when that's possible.
 */

long NFmiGrib::NormalizedGridType() {

  long type;

  if (itsEdition == 1)
    type = itsGridType;

  else {
    switch (itsGridDefinitionTemplate) {
      case 0: // ll
      case 1: // rll
      case 2: // stretched ll
      case 3: // stretched rll
        type = 10 * itsGridDefinitionTemplate;
        break;

      case 20: // polar stereographic
        type = 5;
        break;

      case 30: // lambert conformal
      case 40: // gaussian ll
      	type = itsGridDefinitionTemplate / 10;
        break;

      default:
         type = itsGridDefinitionTemplate;
         break;

    }
  }

  return type;
}

double NFmiGrib::XResolution() {
  return itsXResolution;
}

double NFmiGrib::YResolution() {
  return itsYResolution;
}

/*
 * Clear()
 *
 * This function is every time a new grib message is read. It
 * initializes all grib message specific variables.
 */

void NFmiGrib::Clear() {

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
}

double NFmiGrib::X0() {
  return itsLongitudeOfFirstGridPoint;
}

double NFmiGrib::Y0() {
  return itsLatitudeOfFirstGridPoint;
}

double NFmiGrib::SouthPoleX() {
  return itsLongitudeOfSouthernPole;
}

double NFmiGrib::SouthPoleY() {
  return itsLatitudeOfSouthernPole;
}

long NFmiGrib::Edition() {
  return itsEdition;
}

void NFmiGrib::MultiGribSupport(bool theMultiGribSupport) {
  if (theMultiGribSupport)
    grib_multi_support_on(0); // Multigrib support on
  else
    grib_multi_support_off(0); // Multigrib support on
}
