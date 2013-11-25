/*
 * NFmiGribMessage.cpp
 *
 *  Created on: Oct 16, 2012
 *      Author: partio
 */

#include "NFmiGribMessage.h"
#include <stdexcept>
#include <boost/lexical_cast.hpp>

const long INVALID_INT_VALUE = -999;
const float kFloatMissing = 32700;

#define CHECK_BIT(var,pos) ((var) & (1<<(pos)))

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

  itsLevelTypeMap.insert(element(1,1)); // ground
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

  long t = 0;

  Clear();

  // Edition-specific keys

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

  // Grib values


  if (itsHandle)
    grib_handle_delete(itsHandle);

  itsHandle = grib_handle_clone(h); // have to clone handle since data values are not returned until called

  return true;
}

long NFmiGribMessage::LocalDefinitionNumber() const
{
  
  if (Edition() == 1) {
    long l;
    GRIB_CHECK(grib_get_long(itsHandle, "localDefinitionNumber", &l), 0);
    return l;
  }
  else
  {
    return ProductDefinitionTemplateNumber();
  }
}

long NFmiGribMessage::ProductDefinitionTemplateNumber() const
{
  long l = INVALID_INT_VALUE;

  if (Edition() == 2)
  {
    GRIB_CHECK(grib_get_long(itsHandle,"productDefinitionTemplateNumber",&l),0);
  }

  return l;
}

void NFmiGribMessage::ProductDefinitionTemplateNumber(long theNumber)
{
  GRIB_CHECK(grib_set_long(itsHandle,"productDefinitionTemplateNumber",theNumber),0);
}

long NFmiGribMessage::TypeOfStatisticalProcessing() const
{
  long l = INVALID_INT_VALUE;

  if (Edition() == 2)
  {
    GRIB_CHECK(grib_get_long(itsHandle,"typeOfStatisticalProcessing",&l),0);
  }

  return l;
}

void NFmiGribMessage::TypeOfStatisticalProcessing(long theType)
{
  if (Edition() == 2)
  {
    GRIB_CHECK(grib_set_long(itsHandle,"typeOfStatisticalProcessing",theType),0);
  }
}

long NFmiGribMessage::DataType() const
{
  long l;
  GRIB_CHECK(grib_get_long(itsHandle, "type", &l),0);
  return l;
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

  if (Bitmap()) {
    GRIB_CHECK(grib_set_double(itsHandle,"missingValue",kFloatMissing),0);
  }
  size_t values_len = ValuesLength();
  double* vals = static_cast<double*> (malloc(values_len*sizeof(double)));

  GRIB_CHECK(grib_get_double_array(itsHandle,"values",vals,&values_len),0);

  return vals;
}

void NFmiGribMessage::Values(const double* theValues, long theValuesLength) {

  if (Bitmap()) {
    GRIB_CHECK(grib_set_double(itsHandle,"missingValue",static_cast<double> (kFloatMissing)),0);
  }

  if (Edition() == 2) {
    GRIB_CHECK(grib_set_long(itsHandle,"numberOfValues",theValuesLength),0);
  }

  GRIB_CHECK(grib_set_double_array(itsHandle,"values",theValues,theValuesLength),0);
}

size_t NFmiGribMessage::ValuesLength() const {
  size_t s;
  GRIB_CHECK(grib_get_size(itsHandle,"values",&s),0);
  return s;
}

long NFmiGribMessage::DataDate() const {
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"dataDate",&l),0);
  return l;
}

void NFmiGribMessage::DataDate(long theDate) {
  GRIB_CHECK(grib_set_long(itsHandle,"dataDate",theDate),0);
}

long NFmiGribMessage::DataTime() const {
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"dataTime",&l),0);
  return l;
}

void NFmiGribMessage::DataTime(long theTime) {
  GRIB_CHECK(grib_set_long(itsHandle,"dataTime",theTime),0);
}

long NFmiGribMessage::ForecastTime() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"forecastTime",&l),0);

  return l;
}

void NFmiGribMessage::ForecastTime(long theTime)
{
  GRIB_CHECK(grib_set_long(itsHandle,"forecastTime",theTime),0);
}

std::string NFmiGribMessage::ParameterUnit() const {

  size_t len = 255;
  char unit[256];

  std::string keyName = "units";

  if (Edition() == 2)
  {
    keyName = "parameterUnits";
  }

  GRIB_CHECK(grib_get_string(itsHandle,keyName.c_str(), unit, &len), 0);

  return std::string(unit);
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
  char name[256];

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

double NFmiGribMessage::iDirectionIncrement() const
{
  double d;

  GRIB_CHECK(grib_get_double(itsHandle,"iDirectionIncrementInDegrees",&d),0);

  return d;
}

void NFmiGribMessage::iDirectionIncrement(double theIncrement)
{
  GRIB_CHECK(grib_set_double(itsHandle,"iDirectionIncrementInDegrees",theIncrement),0);
}

double NFmiGribMessage::jDirectionIncrement() const
{
  double d;

  GRIB_CHECK(grib_get_double(itsHandle,"jDirectionIncrementInDegrees",&d),0);

  return d;
}

void NFmiGribMessage::jDirectionIncrement(double theIncrement)
{
  GRIB_CHECK(grib_set_double(itsHandle,"jDirectionIncrementInDegrees",theIncrement),0);
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

/*
 * Clear()
 *
 * This function is every time a new grib message is read. It
 * initializes all grib message specific variables.
 */

void NFmiGribMessage::Clear() {
 
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

  itsPackedValuesLength = INVALID_INT_VALUE;
}

double NFmiGribMessage::X0() const {
  double d;

  GRIB_CHECK(grib_get_double(itsHandle,"longitudeOfFirstGridPointInDegrees",&d),0);

  if (Edition() == 2)
  {
//    d -= 180;
  }

  return d;
}

double NFmiGribMessage::Y0() const {
  double d;

  GRIB_CHECK(grib_get_double(itsHandle,"latitudeOfFirstGridPointInDegrees",&d),0);

  return d;
}

void NFmiGribMessage::X0(double theX0) {
  GRIB_CHECK(grib_set_double(itsHandle,"longitudeOfFirstGridPointInDegrees",theX0),0);
}

void NFmiGribMessage::Y0(double theY0) {
  GRIB_CHECK(grib_set_double(itsHandle,"latitudeOfFirstGridPointInDegrees",theY0),0);
}

double NFmiGribMessage::X1() const {
  double d;

  GRIB_CHECK(grib_get_double(itsHandle,"longitudeOfLastGridPointInDegrees",&d),0);

  if (Edition() == 2)
  {
//    d -= 180;
  }
  
  return d;
}

double NFmiGribMessage::Y1() const {
  double d;

  GRIB_CHECK(grib_get_double(itsHandle,"latitudeOfLastGridPointInDegrees",&d),0);

  return d;
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

void NFmiGribMessage::SouthPoleX(double theLongitude) {
  GRIB_CHECK(grib_set_double(itsHandle,"longitudeOfSouthernPoleInDegrees",theLongitude),0);
}

double NFmiGribMessage::SouthPoleY() const {
  double d;

  GRIB_CHECK(grib_get_double(itsHandle,"latitudeOfSouthernPoleInDegrees",&d),0);
  return d;
}

void NFmiGribMessage::SouthPoleY(double theLatitude) {
  GRIB_CHECK(grib_set_double(itsHandle,"latitudeOfSouthernPoleInDegrees",theLatitude),0);
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

void NFmiGribMessage::IScansNegatively(bool theNegativeIScan) {
  GRIB_CHECK(grib_set_long(itsHandle,"iScansNegatively",static_cast<long> (theNegativeIScan)),0);
}

void NFmiGribMessage::JScansPositively(bool thePositiveJScan) {
  GRIB_CHECK(grib_set_long(itsHandle,"jScansPositively",static_cast<long> (thePositiveJScan)),0);
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

long NFmiGribMessage::NumberOfMissing() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"numberOfMissing",&l),0);

  return l;
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

  for (boost::bimap<long,long>::const_iterator iter = itsLevelTypeMap.begin(), iend = itsLevelTypeMap.end();
      iter != iend; ++iter )
  {
      // iter->left  : grib1
      // iter->right : grib2

    if (targetEdition == 1 && iter->right == levelType)
    {
      return iter->left;
    }
    else if (targetEdition == 2 && iter->left == levelType)
    {
      return iter->right;
    }
  }

  return INVALID_INT_VALUE;

}

long NFmiGribMessage::GridTypeToAnotherEdition(long gridType, long targetEdition) const {

  for (boost::bimap<long,long>::const_iterator iter = itsGridTypeMap.begin(), iend = itsGridTypeMap.end();
    iter != iend; ++iter )
  {
    // iter->left  : grib1
    // iter->right : grib2

    if (targetEdition == 1 && iter->right == gridType)
    {
      return iter->left;
    }
    else if (targetEdition == 2 && iter->left == gridType)
    {
      return iter->right;
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

long NFmiGribMessage::NormalizedStep(bool endStep, bool flatten) const {

  long step = INVALID_INT_VALUE;

  if (Edition() == 1) {

    // http://rda.ucar.edu/docs/formats/grib/gribdoc/timer.html

    long timeRangeIndicator = TimeRangeIndicator();

    switch (timeRangeIndicator)
    {
      // "normal" case
        case 0:
        case 1:
          step = P1();
          break;

        case 10:
          // Old Harmonie
          step = (P1() << 8 ) | P2();
          break;

        case 2:
        case 3:
        case 4:
        case 5:
          if (endStep)
            step = P2();
          else
            step = P1();
          break;

        default:
          timeRangeIndicator = 1; // default to hour
          break;

    }
  }
  else {
    step = ForecastTime();
  }

  if (!flatten)
    return step;

  long multiplier = 1;
  long unitOfTimeRange = NormalizedUnitOfTimeRange();

  // http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-4.shtml
  
  switch (unitOfTimeRange)
  {
	case 0: // minute
    case 1: // hour
	  break;
		
    case 10: // 3 hours
      multiplier = 3;
      break;

    case 11: // 6 hours
      multiplier = 6;
      break;

    case 12: // 12 hours
      multiplier = 12;
      break;

    case 13: // 15 minutes
      multiplier = 15;
      break;

    case 14: // 30 minutes
      multiplier = 30;
      break;

    default:
      break;
  }

  if (step != INVALID_INT_VALUE)
    step *= multiplier;
  
  return step;
}

long NFmiGribMessage::StartStep() const {
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"startStep",&l),0);
  return l;
}

void NFmiGribMessage::StartStep(long theStartStep) {
  GRIB_CHECK(grib_set_long(itsHandle,"startStep",theStartStep),0);
}

long NFmiGribMessage::EndStep() const {
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"endStep",&l),0);
  return l;
}

void NFmiGribMessage::EndStep(long theEndStep) {
  GRIB_CHECK(grib_set_long(itsHandle,"endStep",theEndStep),0);
}

long NFmiGribMessage::StepUnits() const {
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"stepUnits",&l),0);
  return l;
}

void NFmiGribMessage::StepUnits(long theUnit) {
  GRIB_CHECK(grib_set_long(itsHandle,"stepUnits",theUnit),0);
}

long NFmiGribMessage::StepRange() const {
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"stepRange",&l),0);
  return l;
}

void NFmiGribMessage::StepRange(long theRange) {
  GRIB_CHECK(grib_set_long(itsHandle,"stepRange",theRange),0);
}

long NFmiGribMessage::Year() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"year", &l),0);

  return l;
}

void NFmiGribMessage::Year(const std::string& theYear) {
  GRIB_CHECK(grib_set_long(itsHandle,"year",boost::lexical_cast<long> (theYear)),0);
}

long NFmiGribMessage::Month() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"month", &l),0);

  return l;
}

void NFmiGribMessage::Month(const std::string& theMonth) {
  GRIB_CHECK(grib_set_long(itsHandle,"month",boost::lexical_cast<long> (theMonth)),0);
}

long NFmiGribMessage::Day() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"day", &l),0);

  return l;
}

void NFmiGribMessage::Day(const std::string& theDay) {
  GRIB_CHECK(grib_set_long(itsHandle,"day",boost::lexical_cast<long> (theDay)),0);
}

long NFmiGribMessage::Hour() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"hour", &l),0);

  return l;
}

void NFmiGribMessage::Hour(const std::string& theHour) {
  GRIB_CHECK(grib_set_long(itsHandle,"hour",boost::lexical_cast<long> (theHour)),0);
}

long NFmiGribMessage::Minute() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"minute", &l),0);

  return l;
}

void NFmiGribMessage::Minute(const std::string& theMinute) {
  GRIB_CHECK(grib_set_long(itsHandle,"minute",boost::lexical_cast<long> (theMinute)),0);
}

long NFmiGribMessage::Second() const {
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"second", &l),0);

  return l;
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
  if (Edition() == 2) {
    if (theBitmap) {
      GRIB_CHECK(grib_set_long(itsHandle,"bitMapIndicator", 0), 0);
    }
    else {
      GRIB_CHECK(grib_set_long(itsHandle,"bitMapIndicator", 255), 0);
    }
  }
  else {
    GRIB_CHECK(grib_set_long(itsHandle,"bitmapPresent", static_cast<long> (theBitmap)), 0);
  }
}

long NFmiGribMessage::BitsPerValue() const
{
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"bitsPerValue",&l), 0);
  return l;
}

void NFmiGribMessage::BitsPerValue(long theBitsPerValue)
{
  GRIB_CHECK(grib_set_long(itsHandle,"bitsPerValue", theBitsPerValue), 0);
}

bool NFmiGribMessage::UVRelativeToGrid() const
{
  long l = -1;

  if (Edition() == 1)
  {
    GRIB_CHECK(grib_get_long(itsHandle,"uvRelativeToGrid", &l), 0);
  }
  else
  {
   long r = ResolutionAndComponentFlags();

   l = (CHECK_BIT(r, 3) == 8) ? 1 : 0; // in grib2 4th bit tells if uv is relative to grid or not (1000 == 8 == true)
  }
  
  if (l < 0 || l > 1)
  {
    throw std::runtime_error("Unknown value in uvRelativeToGrid(): " + boost::lexical_cast<std::string> (l));
  }

  return static_cast<bool> (l);
}

void NFmiGribMessage::UVRelativeToGrid(bool theRelativity)
{
  if (Edition() == 1)
  {
    GRIB_CHECK(grib_set_long(itsHandle,"uvRelativeToGrid", static_cast<long> (theRelativity)), 0);
  }
}

long NFmiGribMessage::ResolutionAndComponentFlags() const
{
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"resolutionAndComponentFlags", &l), 0);

  return l;
}

void NFmiGribMessage::ResolutionAndComponentFlags(long theResolutionAndComponentFlags)
{
    GRIB_CHECK(grib_set_long(itsHandle,"resolutionAndComponentFlags", theResolutionAndComponentFlags), 0);
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

long NFmiGribMessage::TypeOfGeneratingProcess() const
{
  long l;

  GRIB_CHECK(grib_get_long(itsHandle,"typeOfGeneratingProcess",&l), 0);

  return l;

}

void NFmiGribMessage::TypeOfGeneratingProcess(long theProcess)
{
  GRIB_CHECK(grib_set_long(itsHandle,"typeOfGeneratingProcess",theProcess),0);
}


void NFmiGribMessage::XLengthInMeters(double theLength) {
  GRIB_CHECK(grib_set_double(itsHandle,"DxInMetres",theLength),0);
}

void NFmiGribMessage::YLengthInMeters(double theLength) {
  GRIB_CHECK(grib_set_double(itsHandle,"DyInMetres",theLength),0);
}

double NFmiGribMessage::XLengthInMeters() const {
  double d;
  GRIB_CHECK(grib_get_double(itsHandle,"DxInMetres",&d),0);
  return d;
}

double NFmiGribMessage::YLengthInMeters() const {
  double d;
  GRIB_CHECK(grib_get_double(itsHandle,"DyInMetres",&d),0);
  return d;}

long NFmiGribMessage::NormalizedUnitOfTimeRange() const
{

  // http://www.nco.ncep.noaa.gov/pmb/docs/on388/table4.html
  // http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-4.shtml

  long l;

  if (Edition() == 1) {
    GRIB_CHECK(grib_get_long(itsHandle,"unitOfTimeRange",&l), 0);
  }

  else {
    GRIB_CHECK(grib_get_long(itsHandle,"indicatorOfUnitOfTimeRange",&l), 0);

  switch (l) {

    case 13:
      l = 254; // second
      break;

    default:
      break;
  }
  }

  return l;
}

long NFmiGribMessage::UnitOfTimeRange() const
{
  long l;
  if (Edition() == 1)
    GRIB_CHECK(grib_get_long(itsHandle,"unitOfTimeRange",&l), 0);
  else
    GRIB_CHECK(grib_get_long(itsHandle,"indicatorOfUnitOfTimeRange",&l), 0);
  return l;
}

void NFmiGribMessage::UnitOfTimeRange(long theUnit)
{
  if (Edition() == 1)
    GRIB_CHECK(grib_set_long(itsHandle,"unitOfTimeRange",theUnit),0);
  else
    GRIB_CHECK(grib_set_long(itsHandle,"indicatorOfUnitOfTimeRange",theUnit), 0);
}

long NFmiGribMessage::UnitForTimeRange() const
{
  long l = INVALID_INT_VALUE;

  if (Edition() == 2)
    GRIB_CHECK(grib_get_long(itsHandle,"indicatorOfUnitForTimeRange",&l), 0);
  
  return l;
}

void NFmiGribMessage::UnitForTimeRange(long theUnit)
{
  if (Edition() == 2)
    GRIB_CHECK(grib_set_long(itsHandle,"indicatorOfUnitForTimeRange",theUnit), 0);
}

long NFmiGribMessage::LengthOfTimeRange() const
{
  long l = INVALID_INT_VALUE;

  if (Edition() == 2)
  {
    GRIB_CHECK(grib_get_long(itsHandle,"lengthOfTimeRange",&l), 0);
  }

  return l;
}

void NFmiGribMessage::LengthOfTimeRange(long theLength)
{
  if (Edition() == 2)
  {
    GRIB_CHECK(grib_set_long(itsHandle,"lengthOfTimeRange",theLength),0);
  }
}

long NFmiGribMessage::TimeRangeIndicator() const
{
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"timeRangeIndicator",&l), 0);

  return l;
}

void NFmiGribMessage::TimeRangeIndicator(long theTimeRangeIndicator)
{
  GRIB_CHECK(grib_set_long(itsHandle,"timeRangeIndicator",theTimeRangeIndicator),0);
}

long NFmiGribMessage::P1() const
{
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"P1",&l), 0);

  return l;
}

void NFmiGribMessage::P1(long theP1)
{
  if (Edition() == 1)
    GRIB_CHECK(grib_set_long(itsHandle,"P1",theP1), 0);
}

long NFmiGribMessage::P2() const
{
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"P2",&l), 0);

  return l;
}

void NFmiGribMessage::P2(long theP2)
{
  if (Edition() == 1)
    GRIB_CHECK(grib_set_long(itsHandle,"P2",theP2), 0);
}

long NFmiGribMessage::NV() const
{
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"NV",&l), 0);

  return l;
}

void NFmiGribMessage::NV(long theNV)
{
  GRIB_CHECK(grib_set_long(itsHandle,"NV",theNV), 0);
}

std::vector<double> NFmiGribMessage::PV(size_t theNumberOfCoordinates, size_t level)
{

  double* pv = static_cast<double *> (malloc(theNumberOfCoordinates*sizeof(double)));
  GRIB_CHECK(grib_get_double_array(itsHandle,"pv",pv,&theNumberOfCoordinates),0);
  std::vector<double> ret;

  if (theNumberOfCoordinates == 2) {  /* Hirlam: A, B, A, B */
      ret.push_back(pv[0]);
      ret.push_back(pv[1]);
  }
  else
  {
/* More vertical parameters, let's get'em 
    AROME, ECMWF: A, A, A, B, B, B */
    if (level <= theNumberOfCoordinates)
    {
      ret.push_back((pv[level -1] + pv[level]) / 2.);  
      ret.push_back((pv[theNumberOfCoordinates/2 + level -1] + pv[theNumberOfCoordinates/2 + level]) / 2.); 
    }
  }
  free(pv);
  return ret;
}

void NFmiGribMessage::PV(const std::vector<double>& theAB, size_t abLen)
{
  GRIB_CHECK(grib_set_long(itsHandle,"PVPresent", 1), 0);
  GRIB_CHECK(grib_set_double_array(itsHandle,"pv",&theAB[0],abLen),0);
}

bool NFmiGribMessage::Write(const std::string &theFileName, bool appendToFile) {
  // Assume we have required directory structure in place

  std::string mode = "w";

  if (appendToFile)
    mode = "a";

  GRIB_CHECK(grib_write_message(itsHandle, theFileName.c_str(), mode.c_str()), 0);

  return true;
}

size_t NFmiGribMessage::PackedValuesLength() const
{
  if (itsPackedValuesLength == static_cast<size_t> (INVALID_INT_VALUE))
  {
    long length;
 
    if (Edition() == 1)
    {
      GRIB_CHECK(grib_get_long(itsHandle,"section4Length",&length),0);
      length -= 11;
    }
    else
    {
      GRIB_CHECK(grib_get_long(itsHandle,"section7Length",&length),0);
      length -= 5;
    }

    itsPackedValuesLength = length;
  }

  return itsPackedValuesLength;
}

size_t NFmiGribMessage::BytesLength(const std::string& key) const
{
  size_t s;
  
  GRIB_CHECK(grib_get_size(itsHandle,key.c_str(),&s),0);

  return s;
}

bool NFmiGribMessage::Bytes(const std::string& key, unsigned char* data) const
{
  size_t length = BytesLength(key);
  
  GRIB_CHECK(grib_get_bytes(itsHandle,key.c_str(),data, &length),0);

  return true;
}

bool NFmiGribMessage::PackedValues(unsigned char* data) const
{
#ifdef GRIB_READ_PACKED_DATA
  size_t dataLength;

  GRIB_CHECK(grib_get_packed_values(itsHandle,data,&dataLength),0);

  assert(dataLength == PackedValuesLength());

#else
#warning GRIB_READ_PACKED_DATA not defined -- reading packed data with fmigrib is not supported
  throw std::runtime_error("This version on NFmiGrib is not compiled with support for reading of packed data");
#endif

  return true;
}

double NFmiGribMessage::BinaryScaleFactor() const
{
  double d;
  GRIB_CHECK(grib_get_double(itsHandle,"binaryScaleFactor",&d), 0);

  return d;
}

double NFmiGribMessage::DecimalScaleFactor() const
{
  double d;
  GRIB_CHECK(grib_get_double(itsHandle,"decimalScaleFactor",&d), 0);

  return d;
}

double NFmiGribMessage::ReferenceValue() const
{
  double d;
  GRIB_CHECK(grib_get_double(itsHandle,"referenceValue",&d), 0);

  return d;
}

long NFmiGribMessage::Section4Length() const
{
  long l;
  GRIB_CHECK(grib_get_long(itsHandle,"section4Length",&l), 0);

  return l;
}

bool NFmiGribMessage::KeyExists(const std::string& theKey) const
{
  int i;
  i = grib_is_defined(itsHandle, theKey.c_str());
  return (i == 0 ? false: true);
}
