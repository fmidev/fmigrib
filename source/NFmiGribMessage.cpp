/*
 * NFmiGribMessageMessage.cpp
 *
 *  Created on: Oct 16, 2012
 *      Author: partio
 */

#include "NFmiGribMessage.h"
#include <stdexcept>
#include <boost/lexical_cast.hpp>

const long INVALID_INT_VALUE = -999;
const float kFloatMissing = 32700;

NFmiGribMessage::NFmiGribMessage() {

  itsHandle = grib_handle_new_from_template(NULL,"GRIB2");

  if (!itsHandle)
    throw std::runtime_error("Unable to create grib handle");

  Clear();

  typedef boost::bimap<long,long>::value_type element;

  // GRIB1 <--> GRIB2

  itsGridTypeMap.insert(element(0,0)); // ll
  itsGridTypeMap.insert(element(10,1)); // rll
  itsGridTypeMap.insert(element(20,2)); // stretched ll
  itsGridTypeMap.insert(element(30,3)); // stretched rll
  itsGridTypeMap.insert(element(5,20)); // polar stereographic
  itsGridTypeMap.insert(element(3,30)); // lambert conformal
  itsGridTypeMap.insert(element(4,40)); // gaussian ll

  // GRIB1 <--> GRIB2

  itsLevelTypeMap.insert(element(100,100)); // isobaric
  itsLevelTypeMap.insert(element(160,160)); // depth below sea
  itsLevelTypeMap.insert(element(102,101)); // mean sea
  itsLevelTypeMap.insert(element(103,102)); // specific altitude above mean-sea level
  itsLevelTypeMap.insert(element(105,103)); // specified height above ground
  itsLevelTypeMap.insert(element(109,105)); // hybrid
  itsLevelTypeMap.insert(element(111,106)); // depth below land surface

}

NFmiGribMessage::~NFmiGribMessage() {

  if (itsHandle)
    grib_handle_delete(itsHandle);

}

bool NFmiGribMessage::Read(grib_handle *h) {

  if (itsHandle) {
	  grib_handle_delete(itsHandle);
  }

  itsHandle = grib_handle_clone(h); // have to clone handle since data values are not returned until called

  long t = 0;

  Clear();

  GRIB_CHECK(grib_get_long(h, "totalLength", &itsTotalLength), 0);

  GRIB_CHECK(grib_get_long(h,"dataDate",&itsDate),0);
  GRIB_CHECK(grib_get_long(h,"dataTime",&itsTime),0);


  GRIB_CHECK(grib_get_long(h,"stepUnits",&itsStepUnits),0);
  GRIB_CHECK(grib_get_long(h,"stepRange",&itsStepRange),0);

  GRIB_CHECK(grib_get_long(h, "year", &itsYear), 0);
  GRIB_CHECK(grib_get_long(h, "month", &itsMonth), 0);
  GRIB_CHECK(grib_get_long(h, "day", &itsDay), 0);
  GRIB_CHECK(grib_get_long(h, "hour", &itsHour), 0);
  GRIB_CHECK(grib_get_long(h, "minute", &itsMinute), 0);

  GRIB_CHECK(grib_get_long(h,"timeRangeIndicator", &itsTimeRangeIndicator), 0);

  // Edition-specific keys

  if (Edition() == 1) {

    //GRIB_CHECK(grib_get_long(h,"indicatorOfTypeOfLevel",&itsIndicatorOfTypeOfLevel),0);

    GRIB_CHECK(grib_get_long(h,"startStep",&itsStartStep),0);
    GRIB_CHECK(grib_get_long(h,"endStep",&itsEndStep),0);

    t = 0;

    if (grib_get_long(h, "localDefinitionNumber", &t) == GRIB_SUCCESS)
      itsLocalDefinitionNumber = t;

    t = 0;

    if (grib_get_long(h, "type", &t) == GRIB_SUCCESS)
      itsDataType = t;

  }
  else if (Edition() == 2) {

    //GRIB_CHECK(grib_get_long(h,"typeOfFirstFixedSurface",&itsTypeOfFirstFixedSurface),0);

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

  // Projection-specific keys

  int gridType = NormalizedGridType();

  if (gridType == 0 || gridType == 10) { // latlon or rot latlon
    //GRIB_CHECK(grib_get_double(h,"latitudeOfLastGridPointInDegrees",&itsLatitudeOfLastGridPoint),0);
    //GRIB_CHECK(grib_get_double(h,"longitudeOfLastGridPointInDegrees",&itsLongitudeOfLastGridPoint),0);

    GRIB_CHECK(grib_get_double(h,"iDirectionIncrementInDegrees",&itsXResolution),0);
    GRIB_CHECK(grib_get_double(h,"jDirectionIncrementInDegrees",&itsYResolution),0);

    if (gridType == 10) {
      //GRIB_CHECK(grib_get_double(h,"latitudeOfSouthernPoleInDegrees",&itsLatitudeOfSouthernPole),0);
      //GRIB_CHECK(grib_get_double(h,"longitudeOfSouthernPoleInDegrees",&itsLongitudeOfSouthernPole),0);
    }
  }
  else if (gridType == 5) { // (polar) Stereographic
    //GRIB_CHECK(grib_get_double(h,"orientationOfTheGrid",&itsOrientationOfTheGrid),0);
/*
    GRIB_CHECK(grib_get_double(h,"yDirectionGridLengthInMetres",&itsYResolution),0);
    GRIB_CHECK(grib_get_double(h,"xDirectionGridLengthInMetres",&itsXResolution),0);
*/
  }
  else
    throw std::runtime_error("Unsupported projection");


  // Grib values

  GRIB_CHECK(grib_get_size(h,"values",&itsValuesLength),0);

  if (itsHandle)
    grib_handle_delete(itsHandle);

  itsHandle = grib_handle_clone(h); // have to clone handle since data values are not returned until called

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

  // Set missing value to kFloatMissing

  if (Bitmap())
    GRIB_CHECK(grib_set_double(itsHandle,"missingValue",kFloatMissing),0);

  double* vals = static_cast<double*> (malloc(itsValuesLength*sizeof(double)));

  GRIB_CHECK(grib_get_double_array(itsHandle,"values",vals,&itsValuesLength),0);

  return vals;
}

void NFmiGribMessage::Values(const double* theValues, long theValuesLength) {

  if (Bitmap())
    GRIB_CHECK(grib_set_double(itsHandle,"missingValue",static_cast<double> (kFloatMissing)),0);

  GRIB_CHECK(grib_set_long(itsHandle,"numberOfValues",theValuesLength),0);
  GRIB_CHECK(grib_set_double_array(itsHandle,"values",theValues,theValuesLength),0);
}

int NFmiGribMessage::ValuesLength() const {
  return static_cast<int> (itsValuesLength);
}

long NFmiGribMessage::DataDate() const {
  return itsDate;
}

long NFmiGribMessage::DataTime() const {
  return itsTime;
}

long NFmiGribMessage::ForecastTime() const {
  return itsForecastTime;
}

long NFmiGribMessage::ParameterNumber() const {
  long l;

  if (Edition() == 1) {
    GRIB_CHECK(grib_get_long(itsHandle,"indicatorOfParameter",&l),0);
  }
  else {
    GRIB_CHECK(grib_get_long(itsHandle,"parameterNumber",&l),0);
  }

  return l;
}

long NFmiGribMessage::ParameterCategory() const {
  if (Edition() == 2) {
	  long l;
      GRIB_CHECK(grib_get_long(itsHandle,"parameterCategory",&l),0);
      return l;
  }
  else
    return INVALID_INT_VALUE;
}

long NFmiGribMessage::ParameterDiscipline() const {
  if (Edition() == 2) {
    long l;
    GRIB_CHECK(grib_get_long(itsHandle,"discipline",&l),0);
    return l;
  }
  else
    return INVALID_INT_VALUE;
}

void NFmiGribMessage::ParameterNumber(long theNumber) {

  if (Edition() == 1) {
    GRIB_CHECK(grib_set_long(itsHandle,"indicatorOfParameter",theNumber),0);
  }
  else {
    GRIB_CHECK(grib_set_long(itsHandle,"parameterNumber",theNumber),0);
  }
}

void NFmiGribMessage::ParameterCategory(long theCategory) {
  GRIB_CHECK(grib_set_long(itsHandle,"parameterCategory",theCategory),0);
}

void NFmiGribMessage::ParameterDiscipline(long theDiscipline) {
  GRIB_CHECK(grib_set_long(itsHandle,"discipline",theDiscipline),0);
}

std::string NFmiGribMessage::ParameterName() const {
  size_t len = 255;
  char name[1024];

  GRIB_CHECK(grib_get_string(itsHandle, "parameterName", name, &len), 0);

  return std::string(name);
}

long NFmiGribMessage::GridType() const {
  long l;

  if (Edition() == 1)
    GRIB_CHECK(grib_get_long(itsHandle,"dataRepresentationType",&l),0);
  else
    GRIB_CHECK(grib_get_long(itsHandle,"gridDefinitionTemplateNumber",&l),0);

  return l;
}

void NFmiGribMessage::GridType(long theGridType) {
  if (Edition() == 1)
    GRIB_CHECK(grib_set_long(itsHandle,"dataRepresentationType",theGridType),0);
  else
    GRIB_CHECK(grib_set_long(itsHandle,"gridDefinitionTemplateNumber",theGridType),0);
}

double NFmiGribMessage::GridOrientation() const {
  double d;
  GRIB_CHECK(grib_get_double(itsHandle,"orientationOfTheGridInDegrees",&d),0);
  return d;
}

void NFmiGribMessage::GridOrientation(double theOrientation) {
  GRIB_CHECK(grib_set_double(itsHandle,"orientationOfTheGridInDegrees",theOrientation),0);
}

long NFmiGribMessage::LevelType() const {
  long l;

  if (Edition() == 2)
    GRIB_CHECK(grib_get_long(itsHandle,"typeOfFirstFixedSurface",&l),0);

  else
    GRIB_CHECK(grib_get_long(itsHandle,"indicatorOfTypeOfLevel",&l),0);

  return l;
}

void NFmiGribMessage::LevelType(long theLevelType) {
  if (Edition() == 2)
    GRIB_CHECK(grib_set_long(itsHandle,"typeOfFirstFixedSurface",theLevelType),0);

  else
    GRIB_CHECK(grib_set_long(itsHandle,"indicatorOfTypeOfLevel",theLevelType),0);

}

long NFmiGribMessage::LevelValue() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"level",&l),0);

  return l;
}

void NFmiGribMessage::LevelValue(long theLevelValue) {
  GRIB_CHECK(grib_set_long(itsHandle,"level",theLevelValue),0);
}

double NFmiGribMessage::XResolution() const {
  return itsXResolution;
}

double NFmiGribMessage::YResolution() const {
  return itsYResolution;
}

/*
 * Clear()
 *
 * This function is every time a new grib message is read. It
 * initializes all grib message specific variables.
 */

void NFmiGribMessage::Clear() {

  itsValuesLength = 0;

  itsDate = INVALID_INT_VALUE;
  itsTime = INVALID_INT_VALUE;

  itsValuesLength = INVALID_INT_VALUE;

  itsXResolution = INVALID_INT_VALUE;
  itsYResolution = INVALID_INT_VALUE;

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

double NFmiGribMessage::X0() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"longitudeOfFirstGridPointInDegrees",&l),0);

  return l;
}

double NFmiGribMessage::Y0() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"latitudeOfFirstGridPointInDegrees",&l),0);

  return l;
}

void NFmiGribMessage::X0(double theX0) {
  GRIB_CHECK(grib_set_double(itsHandle,"longitudeOfFirstGridPointInDegrees",theX0),0);
}

void NFmiGribMessage::Y0(double theY0) {
  GRIB_CHECK(grib_set_double(itsHandle,"latitudeOfFirstGridPointInDegrees",theY0),0);
}

double NFmiGribMessage::X1() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"longitudeOfLastGridPointInDegrees",&l),0);

  return l;
}

double NFmiGribMessage::Y1() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"latitudeOfLastGridPointInDegrees",&l),0);

  return l;
}

void NFmiGribMessage::X1(double theX1) {
  GRIB_CHECK(grib_set_double(itsHandle,"longitudeOfLastGridPointInDegrees",theX1),0);
}

void NFmiGribMessage::Y1(double theY1) {
  GRIB_CHECK(grib_set_double(itsHandle,"latitudeOfLastGridPointInDegrees",theY1),0);
}

double NFmiGribMessage::SouthPoleX() const {
  double d;

  GRIB_CHECK(grib_get_double(itsHandle,"longitudeOfSouthernPoleInDegrees",&d),0);
  return d;
}

double NFmiGribMessage::SouthPoleY() const {
  double d;

  GRIB_CHECK(grib_get_double(itsHandle,"latitudeOfSouthernPoleInDegrees",&d),0);
  return d;
}

long NFmiGribMessage::Edition() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"editionNumber",&l),0);

  return l;
}

void NFmiGribMessage::Edition(long theEdition) {
  GRIB_CHECK(grib_set_long(itsHandle,"editionNumber",theEdition),0);
}

long NFmiGribMessage::Process() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"generatingProcessIdentifier",&l),0);

  return l;
}

void NFmiGribMessage::Process(long theProcess) {
  GRIB_CHECK(grib_set_long(itsHandle,"generatingProcessIdentifier",theProcess),0);
}

long NFmiGribMessage::Centre() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"centre",&l),0);

  return l;
}

void NFmiGribMessage::Centre(long theCentre) {
	GRIB_CHECK(grib_set_long(itsHandle,"centre",theCentre),0);
}


bool NFmiGribMessage::IScansNegatively() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"iScansNegatively",&l),0);

  return (l == 1);
}

bool NFmiGribMessage::JScansPositively() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"jScansPositively",&l),0);

  return (l == 1);
}

long NFmiGribMessage::SizeX() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"Ni",&l),0);

  return l;
}

long NFmiGribMessage::SizeY() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"Nj",&l),0);
  return l;
}

void NFmiGribMessage::SizeX(long theXSize) {
  GRIB_CHECK(grib_set_long(itsHandle, "Ni", theXSize), 0);
}

void NFmiGribMessage::SizeY(long theYSize) {
  GRIB_CHECK(grib_set_long(itsHandle, "Nj", theYSize), 0);
}

long NFmiGribMessage::Table2Version() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"table2Version",&l),0);
  return l;
}

void NFmiGribMessage::Table2Version(long theVersion) {
  GRIB_CHECK(grib_set_long(itsHandle, "table2Version", theVersion), 0);
}

/*
 * NormalizedLevelType()
 *
 * Return level type number thats normalized to grib 1 definitions,
 * when that's possible.
 */

long NFmiGribMessage::NormalizedLevelType(unsigned int targetEdition) const {

  long rawtype = LevelType();

  if (Edition() == targetEdition) {
    return rawtype;
  }
  else {
    return LevelTypeToAnotherEdition(rawtype, targetEdition);
  }

}

long NFmiGribMessage::LevelTypeToAnotherEdition(long levelType, long targetEdition) const {

  size_t i = 0;

  if (targetEdition == 1)  {
    while (i < itsLevelTypeMap.size()) {
      if (levelType == itsLevelTypeMap.left.at(i)) {
			  return itsLevelTypeMap.right.at(i);
      }
      i++;
    }
  }
  else if (targetEdition == 2) {
    while (i < itsLevelTypeMap.size()) {
      if (levelType == itsLevelTypeMap.right.at(i)) {
        return itsLevelTypeMap.left.at(i);
      }
      i++;
    }
  }

  return INVALID_INT_VALUE;

}

long NFmiGribMessage::GridTypeToAnotherEdition(long gridType, long targetEdition) const {

  size_t i = 0;

  if (targetEdition == 1)  {
    while (i < itsGridTypeMap.size()) {
      if (gridType == itsGridTypeMap.left.at(i)) {
			  return itsGridTypeMap.right.at(i);
      }
      i++;
    }
  }
  else if (targetEdition == 2) {
    while (i < itsGridTypeMap.size()) {
      if (gridType == itsGridTypeMap.right.at(i)) {
        return itsGridTypeMap.left.at(i);
      }
      i++;
    }
  }

  return INVALID_INT_VALUE;

}



/*
 * NormalizedGridType()
 *
 * Return geom type number thats normalized to grib 1 definitions,
 * when that's possible.
 */

long NFmiGribMessage::NormalizedGridType(unsigned int targetEdition) const {

  long rawtype = GridType();

  if (Edition() == targetEdition) {
    return rawtype;
  }
  else {
    return GridTypeToAnotherEdition(rawtype, targetEdition);
  }

}

void NFmiGribMessage::StartStep(long theStartStep) {
  GRIB_CHECK(grib_set_long(itsHandle,"startStep",theStartStep),0);
}

void NFmiGribMessage::EndStep(long theEndStep) {
  GRIB_CHECK(grib_set_long(itsHandle,"endStep",theEndStep),0);
}

void NFmiGribMessage::Year(const std::string& theYear) {
  GRIB_CHECK(grib_set_long(itsHandle,"year",boost::lexical_cast<long> (theYear)),0);
}

void NFmiGribMessage::Month(const std::string& theMonth) {
  GRIB_CHECK(grib_set_long(itsHandle,"month",boost::lexical_cast<long> (theMonth)),0);
}

void NFmiGribMessage::Day(const std::string& theDay) {
  GRIB_CHECK(grib_set_long(itsHandle,"day",boost::lexical_cast<long> (theDay)),0);
}

void NFmiGribMessage::Hour(const std::string& theHour) {
  GRIB_CHECK(grib_set_long(itsHandle,"hour",boost::lexical_cast<long> (theHour)),0);
}

void NFmiGribMessage::Minute(const std::string& theMinute) {
  GRIB_CHECK(grib_set_long(itsHandle,"minute",boost::lexical_cast<long> (theMinute)),0);
}

void NFmiGribMessage::Second(const std::string& theSecond) {
  GRIB_CHECK(grib_set_long(itsHandle,"second",boost::lexical_cast<long> (theSecond)),0);
}

bool NFmiGribMessage::Bitmap() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"bitmapPresent",&l), 0);

  return static_cast<bool> (l);
}

void NFmiGribMessage::Bitmap(bool theBitmap) {
  //GRIB_CHECK(grib_set_long(itsHandle,"bitmapPresent",static_cast<int> (theBitmap)), 0);
  if (theBitmap)
    GRIB_CHECK(grib_set_long(itsHandle,"bitMapIndicator", 0), 0);
  else
    GRIB_CHECK(grib_set_long(itsHandle,"bitMapIndicator", 255), 0);
}

void NFmiGribMessage::PackingType(const std::string& thePackingType)
{
  // Should probably check edition -- v2 has more packing types

  size_t len = thePackingType.length();
  GRIB_CHECK(grib_set_string(itsHandle, "packingType", thePackingType.c_str(), &len), 0);

}

std::string NFmiGribMessage::PackingType() const
{
  size_t len = 255;
  char type[1024];

  GRIB_CHECK(grib_get_string(itsHandle, "packingType", type, &len), 0);

  return std::string(type);

}

void NFmiGribMessage::XLengthInMeters(double theLength) {
  GRIB_CHECK(grib_set_double(itsHandle,"xDirectionGridLengthInMetres",theLength),0);
}

void NFmiGribMessage::YLengthInMeters(double theLength) {
  GRIB_CHECK(grib_set_double(itsHandle,"yDirectionGridLengthInMetres",theLength),0);
}

bool NFmiGribMessage::Write(const std::string &theFileName) {
  // Assume we have required directory structure in place

  GRIB_CHECK(grib_write_message(itsHandle, theFileName.c_str(), "w"), 0);

  return true;
}
