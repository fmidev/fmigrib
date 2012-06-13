#include "NFmiGrib.h"
#include <stdexcept>

// Should use const int ???

#define INVALID_VALUE -999

NFmiGrib::NFmiGrib() :
  h(0),
  f(0),
  itsMessageCount(-1),
  itsCurrentMessage(-1) {}

NFmiGrib::NFmiGrib(const std::string &theFileName) :
  h(0),
  f(0),
  itsCurrentMessage(-1),
  itsDataLength(-1) {

  Open(theFileName);
}

NFmiGrib::~NFmiGrib() {

  if (h)
    grib_handle_delete(h);

  if (f)
    fclose(f);
}

bool NFmiGrib::Open(const std::string &theFileName) {

  int err = 0;

  // Open with 'rb', although in linux it equals to 'r'

  if (f)
    fclose(f);

  if (!(f = fopen(theFileName.c_str(), "rb")))
    return false;

  if (h)
    grib_handle_delete(h);

  grib_multi_support_on(0);

  if (grib_count_in_file(0, f, &itsMessageCount) != GRIB_SUCCESS) {
    return false;
  }

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

  GRIB_CHECK(grib_get_double(h,"latitudeOfLastGridPointInDegrees",&itsLatitudeOfLastGridPoint),0);
  GRIB_CHECK(grib_get_double(h,"longitudeOfLastGridPointInDegrees",&itsLongitudeOfLastGridPoint),0);

  GRIB_CHECK(grib_get_double(h,"latitudeOfSouthernPoleInDegrees",&itsLatitudeOfSouthernPole),0);
  GRIB_CHECK(grib_get_double(h,"longitudeOfSouthernPoleInDegrees",&itsLongitudeOfSouthernPole),0);

  GRIB_CHECK(grib_get_long(h,"stepUnits",&itsStepUnits),0);
  GRIB_CHECK(grib_get_long(h,"stepRange",&itsStepRange),0);

  GRIB_CHECK(grib_get_long(h,"level",&itsLevel),0);

  GRIB_CHECK(grib_get_double(h,"iDirectionIncrementInDegrees",&itsXResolution),0);
  GRIB_CHECK(grib_get_double(h,"jDirectionIncrementInDegrees",&itsYResolution),0);

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

  }
  else 
    throw std::runtime_error("Unknown grib edition");

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

  GRIB_CHECK(grib_get_size(h,"itsValues",&itsValuesLength),0);

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
    return INVALID_VALUE;
}

long NFmiGrib::ParameterDiscipline() {
  if (itsEdition == 2)
    return itsParameterDiscipline;
  else
    return INVALID_VALUE;
}

long NFmiGrib::GridType() {
  if (itsEdition == 1)
    return itsGridType;
  else
    return itsGridDefinitionTemplate;
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

  itsEdition = INVALID_VALUE;
  itsProcess = INVALID_VALUE;
  itsCentre = INVALID_VALUE;

  itsXSize = INVALID_VALUE;
  itsYSize = INVALID_VALUE;

  itsLatitudeOfFirstGridPoint = INVALID_VALUE;
  itsLongitudeOfFirstGridPoint = INVALID_VALUE;
  itsLatitudeOfLastGridPoint = INVALID_VALUE;
  itsLongitudeOfLastGridPoint = INVALID_VALUE;

  itsIScansNegatively = INVALID_VALUE;
  itsJScansPositively = INVALID_VALUE;

  itsLatitudeOfSouthernPole = INVALID_VALUE;
  itsLongitudeOfSouthernPole = INVALID_VALUE;

  itsIndicatorOfParameter = INVALID_VALUE;
  itsIndicatorOfTypeOfLevel = INVALID_VALUE;

  itsLevel = INVALID_VALUE;
  itsDate = INVALID_VALUE;
  itsTime = INVALID_VALUE;

  itsStepUnits = INVALID_VALUE;
  itsStepRange = INVALID_VALUE;
  itsStartStep = INVALID_VALUE;
  itsEndStep = INVALID_VALUE;

  itsGridType = INVALID_VALUE;
  itsGridDefinitionTemplate = INVALID_VALUE;

  itsValuesLength = INVALID_VALUE;

  itsParameterDiscipline = INVALID_VALUE;
  itsParameterCategory = INVALID_VALUE;
  itsParameterNumber = INVALID_VALUE;
  itsTypeOfFirstFixedSurface = INVALID_VALUE;

  itsXResolution = INVALID_VALUE;
  itsYResolution = INVALID_VALUE;

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
