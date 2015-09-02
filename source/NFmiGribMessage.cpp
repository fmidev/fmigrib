/*
 * NFmiGribMessage.cpp
 *
 *  Created on: Oct 16, 2012
 *      Author: partio
 */

#include "NFmiGribMessage.h"
#include <stdexcept>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include "NFmiGribPacking.h"
#include <cassert>
#include <fstream>

#include <boost/filesystem/path.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/stream.hpp>

const long INVALID_INT_VALUE = -999;
const float kFloatMissing = 32700;

#define CHECK_BIT(var,pos) ((var) & (1<<(pos)))

typedef boost::bimap<long,long>::value_type element;

void NFmiGribMessage::InitMaps()
{  
  if (itsGridTypeMap.size()) return;

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
  itsLevelTypeMap.insert(element(8,8)); // top of atmosphere
  itsLevelTypeMap.insert(element(100,100)); // isobaric
  itsLevelTypeMap.insert(element(160,160)); // depth below sea
  itsLevelTypeMap.insert(element(102,101)); // mean sea
  itsLevelTypeMap.insert(element(103,102)); // specific altitude above mean-sea level
  itsLevelTypeMap.insert(element(105,103)); // specified height above ground
  itsLevelTypeMap.insert(element(109,105)); // hybrid
  itsLevelTypeMap.insert(element(111,106)); // depth below land surface

}

NFmiGribMessage::NFmiGribMessage() 
  : itsHandle(0) {

  Clear();
  InitMaps();
}

NFmiGribMessage::~NFmiGribMessage() {}

NFmiGribMessage::NFmiGribMessage(const NFmiGribMessage& other) 
  : itsHandle(0)
{
  Clear();
  InitMaps();

  if (other.itsHandle)
    itsHandle = grib_handle_clone(other.itsHandle);
}

bool NFmiGribMessage::Read(grib_handle *h) {
  itsHandle = h;
  Clear();
  
  return true;
}

void NFmiGribMessage::Clear()
{
  itsPackedValuesLength = INVALID_INT_VALUE;
  itsEdition = INVALID_INT_VALUE;  
}

void NFmiGribMessage::PerturbationNumber(long thePerturbationNumber)
{
  SetLongKey("perturbationNumber",thePerturbationNumber);
}

long NFmiGribMessage::PerturbationNumber() const
{
  return GetLongKey("perturbationNumber");
}

long NFmiGribMessage::LocalDefinitionNumber() const
{
  
  if (Edition() == 1) {
    return GetLongKey("localDefinitionNumber");
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
    l = GetLongKey("productDefinitionTemplateNumber");
  }

  return l;
}

void NFmiGribMessage::ProductDefinitionTemplateNumber(long theNumber)
{
  SetLongKey("productDefinitionTemplateNumber",theNumber);
}

long NFmiGribMessage::TypeOfStatisticalProcessing() const
{
  long l = INVALID_INT_VALUE;

  if (Edition() == 2)
  {
    l = GetLongKey("typeOfStatisticalProcessing");
  }

  return l;
}

void NFmiGribMessage::TypeOfStatisticalProcessing(long theType)
{
  if (Edition() == 2)
  {
    SetLongKey("typeOfStatisticalProcessing",theType);
  }
}
/*
long NFmiGribMessage::DataType() const
{
  return GetLongKey("type");
}
*/
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
  assert(itsHandle);
  if (Bitmap()) {
    double missingValue = GetDoubleKey("missingValue");

    if (missingValue == 9999) SetDoubleKey("missingValue", kFloatMissing);
  }
  size_t values_len = ValuesLength();
  double* vals = static_cast<double*> (malloc(values_len*sizeof(double)));

  GRIB_CHECK(grib_get_double_array(itsHandle,"values",vals,&values_len),0);

  return vals;
}

void NFmiGribMessage::Values(const double* theValues, long theValuesLength) {

  assert(itsHandle);
  if (Bitmap()) {
    double missingValue = GetDoubleKey("missingValue");

    if (missingValue == 9999) SetDoubleKey("missingValue",kFloatMissing);
  }

  if (Edition() == 2) {
    SetLongKey("numberOfValues",theValuesLength);
  }

  GRIB_CHECK(grib_set_double_array(itsHandle,"values",theValues,theValuesLength),0);
}

size_t NFmiGribMessage::ValuesLength() const {
  return GetSizeTKey("values");
}

long NFmiGribMessage::DataDate() const {
  return GetLongKey("dataDate");
}

void NFmiGribMessage::DataDate(long theDate) {
  SetLongKey("dataDate",theDate);
}

long NFmiGribMessage::DataTime() const {
  return GetLongKey("dataTime");
}

void NFmiGribMessage::DataTime(long theTime) {
  SetLongKey("dataTime",theTime);
}

long NFmiGribMessage::ForecastTime() const {
  return GetLongKey("forecastTime");
}

void NFmiGribMessage::ForecastTime(long theTime)
{
  SetLongKey("forecastTime",theTime);
}

std::string NFmiGribMessage::ParameterUnit() const {

  std::string keyName = "units";

  if (Edition() == 2)
  {
    keyName = "parameterUnits";
  }

  return GetStringKey(keyName);
}

long NFmiGribMessage::ParameterNumber() const {
  if (Edition() == 1) {
    return GetLongKey("indicatorOfParameter");
  }
  else {
    return GetLongKey("parameterNumber");
  }
}

long NFmiGribMessage::ParameterCategory() const {
  long l = INVALID_INT_VALUE;
  if (Edition() == 2) {
    l = GetLongKey("parameterCategory");
  }
  
  return l;
}

long NFmiGribMessage::ParameterDiscipline() const {
  long l = INVALID_INT_VALUE;

  if (Edition() == 2) {
    l = GetLongKey("discipline");
  }
  
  return l;
}

void NFmiGribMessage::ParameterNumber(long theNumber) {

  if (Edition() == 1) {
    SetLongKey("indicatorOfParameter",theNumber);
  }
  else {
    SetLongKey("parameterNumber",theNumber);
  }
}

void NFmiGribMessage::ParameterCategory(long theCategory) {
  if (Edition() == 2)
    SetLongKey("parameterCategory",theCategory);
}

void NFmiGribMessage::ParameterDiscipline(long theDiscipline) {
  if (Edition() == 2)
    SetLongKey("discipline",theDiscipline);
}

std::string NFmiGribMessage::ParameterName() const {
  return GetStringKey("name");
}

long NFmiGribMessage::GridType() const {
  if (Edition() == 1)
    return GetLongKey("dataRepresentationType");
  else
    return GetLongKey("gridDefinitionTemplateNumber");

}

void NFmiGribMessage::GridType(long theGridType) {
  if (Edition() == 1)
    SetLongKey("dataRepresentationType",theGridType);
  else
    SetLongKey("gridDefinitionTemplateNumber",theGridType);
}

double NFmiGribMessage::GridOrientation() const {
  return GetDoubleKey("orientationOfTheGridInDegrees");
}

void NFmiGribMessage::GridOrientation(double theOrientation) {
  SetDoubleKey("orientationOfTheGridInDegrees",theOrientation);
}

double NFmiGribMessage::iDirectionIncrement() const
{
  return GetDoubleKey("iDirectionIncrementInDegrees");
}

void NFmiGribMessage::iDirectionIncrement(double theIncrement)
{
  SetDoubleKey("iDirectionIncrementInDegrees",theIncrement);
}

double NFmiGribMessage::jDirectionIncrement() const
{
  return GetDoubleKey("jDirectionIncrementInDegrees");
}

void NFmiGribMessage::jDirectionIncrement(double theIncrement)
{
  SetDoubleKey("jDirectionIncrementInDegrees",theIncrement);
}

long NFmiGribMessage::LevelType() const {
  if (Edition() == 2)
    return GetLongKey("typeOfFirstFixedSurface");

  else
    return GetLongKey("indicatorOfTypeOfLevel");
}

void NFmiGribMessage::LevelType(long theLevelType) {
  if (Edition() == 2)
    SetLongKey("typeOfFirstFixedSurface",theLevelType);

  else
    SetLongKey("indicatorOfTypeOfLevel",theLevelType);

}

long NFmiGribMessage::LevelValue() const {
  return GetLongKey("level");
}

void NFmiGribMessage::LevelValue(long theLevelValue) {
  if (Edition() == 2)
  {
    SetLongKey("scaleFactorOfFirstFixedSurface", 0);
  SetLongKey("scaledValueOfFirstFixedSurface",theLevelValue);
  }
  else
    SetLongKey("level",theLevelValue);
    
}

double NFmiGribMessage::X0() const {
  return GetDoubleKey("longitudeOfFirstGridPointInDegrees");
}

double NFmiGribMessage::Y0() const {
  return GetDoubleKey("latitudeOfFirstGridPointInDegrees");
}

void NFmiGribMessage::X0(double theX0) {
  SetDoubleKey("longitudeOfFirstGridPointInDegrees",theX0);
}

void NFmiGribMessage::Y0(double theY0) {
  SetDoubleKey("latitudeOfFirstGridPointInDegrees",theY0);
}

double NFmiGribMessage::X1() const {
  return GetDoubleKey("longitudeOfLastGridPointInDegrees");
}

double NFmiGribMessage::Y1() const {
  return GetDoubleKey("latitudeOfLastGridPointInDegrees");
}

void NFmiGribMessage::X1(double theX1) {
  SetDoubleKey("longitudeOfLastGridPointInDegrees",theX1);
}

void NFmiGribMessage::Y1(double theY1) {
  SetDoubleKey("latitudeOfLastGridPointInDegrees",theY1);
}

double NFmiGribMessage::SouthPoleX() const {
  return GetDoubleKey("longitudeOfSouthernPoleInDegrees");
}

void NFmiGribMessage::SouthPoleX(double theLongitude) {
  SetDoubleKey("longitudeOfSouthernPoleInDegrees",theLongitude);
}

double NFmiGribMessage::SouthPoleY() const {
  return GetDoubleKey("latitudeOfSouthernPoleInDegrees");
}

void NFmiGribMessage::SouthPoleY(double theLatitude) {
  SetDoubleKey("latitudeOfSouthernPoleInDegrees",theLatitude);
}

long NFmiGribMessage::Edition() const {
  if (itsEdition != INVALID_INT_VALUE)
  {
    return itsEdition;
  }

  itsEdition = GetLongKey("editionNumber");
  
  return itsEdition;
}

void NFmiGribMessage::Edition(long theEdition) {
  SetLongKey("editionNumber",theEdition);
  itsEdition = theEdition;
}

long NFmiGribMessage::Process() const {
  return GetLongKey("generatingProcessIdentifier");
}

void NFmiGribMessage::Process(long theProcess) {
  SetLongKey("generatingProcessIdentifier",theProcess);
}

long NFmiGribMessage::Centre() const {
  return GetLongKey("centre");
}

void NFmiGribMessage::Centre(long theCentre) {
  SetLongKey("centre",theCentre);
}


bool NFmiGribMessage::IScansNegatively() const {
  return (1 == GetLongKey("iScansNegatively"));
}

bool NFmiGribMessage::JScansPositively() const {
  return (1 == GetLongKey("jScansPositively"));
}

void NFmiGribMessage::IScansNegatively(bool theNegativeIScan) {
  SetLongKey("iScansNegatively",static_cast<long> (theNegativeIScan));
}

void NFmiGribMessage::JScansPositively(bool thePositiveJScan) {
  SetLongKey("jScansPositively",static_cast<long> (thePositiveJScan));
}

long NFmiGribMessage::SizeX() const {
  return GetLongKey("Ni");
}

long NFmiGribMessage::SizeY() const {
  return GetLongKey("Nj");
}

void NFmiGribMessage::SizeX(long theXSize) {
  SetLongKey("Ni", theXSize);
}

void NFmiGribMessage::SizeY(long theYSize) {
  SetLongKey("Nj", theYSize);
}

long NFmiGribMessage::Table2Version() const {
  return GetLongKey("table2Version");
}

void NFmiGribMessage::Table2Version(long theVersion) {
  SetLongKey("table2Version", theVersion);
}

long NFmiGribMessage::NumberOfMissing() const {
  return GetLongKey("numberOfMissing");
}

double NFmiGribMessage::MissingValue() const {
  return GetDoubleKey("missingValue");
}

void NFmiGribMessage::MissingValue(double missingValue) {
  SetDoubleKey("missingValue", missingValue);
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
    if  (ProductDefinitionTemplateNumber() == 8) {
      step = EndStep();
    }
    else {
      step = ForecastTime();
    }
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
  return GetLongKey("startStep");
}

void NFmiGribMessage::StartStep(long theStartStep) {
  SetLongKey("startStep", theStartStep);
}

long NFmiGribMessage::EndStep() const {
  return GetLongKey("endStep");
}

void NFmiGribMessage::EndStep(long theEndStep) {
  SetLongKey("endStep",theEndStep);
}

long NFmiGribMessage::StepUnits() const {
  return GetLongKey("stepUnits");
}

void NFmiGribMessage::StepUnits(long theUnit) {
  SetLongKey("stepUnits",theUnit);
}

long NFmiGribMessage::StepRange() const {
  return GetLongKey("stepRange");
}

void NFmiGribMessage::StepRange(long theRange) {
  SetLongKey("stepRange",theRange);
}

long NFmiGribMessage::Year() const {
  return GetLongKey("year");
}

void NFmiGribMessage::Year(const std::string& theYear) {
  SetLongKey("year", boost::lexical_cast<long> (theYear));
}

long NFmiGribMessage::Month() const {
  return GetLongKey("month");
}

void NFmiGribMessage::Month(const std::string& theMonth) {
  SetLongKey("month", boost::lexical_cast<long> (theMonth));
}

long NFmiGribMessage::Day() const {
  return GetLongKey("day");
}

void NFmiGribMessage::Day(const std::string& theDay) {
  SetLongKey("day", boost::lexical_cast<long> (theDay));
}

long NFmiGribMessage::Hour() const {
  return GetLongKey("hour");
}

void NFmiGribMessage::Hour(const std::string& theHour) {
  SetLongKey("hour", boost::lexical_cast<long> (theHour));
}

long NFmiGribMessage::Minute() const {
  return GetLongKey("minute");
}

void NFmiGribMessage::Minute(const std::string& theMinute) {
  SetLongKey("minute", boost::lexical_cast<long> (theMinute));
}

long NFmiGribMessage::Second() const {
  return GetLongKey("second");
}

void NFmiGribMessage::Second(const std::string& theSecond) {
  SetLongKey("second", boost::lexical_cast<long> (theSecond));
}

bool NFmiGribMessage::Bitmap() const {
  return static_cast<bool> (GetLongKey("bitmapPresent"));
}

void NFmiGribMessage::Bitmap(bool theBitmap) {
  if (Edition() == 2) {
    if (theBitmap) {
      SetLongKey("bitMapIndicator", 0);
    }
    else {
      SetLongKey("bitMapIndicator", 255);
    }
  }
  else {
    SetLongKey("bitmapPresent", static_cast<long> (theBitmap));
  }
}

long NFmiGribMessage::BitsPerValue() const
{
  return GetLongKey("bitsPerValue");
}

void NFmiGribMessage::BitsPerValue(long theBitsPerValue)
{
  SetLongKey("bitsPerValue", theBitsPerValue);
}

bool NFmiGribMessage::UVRelativeToGrid() const
{
  long l = -1;

  if (Edition() == 1)
  {
    l = GetLongKey("uvRelativeToGrid");
  }
  else
  {
   long r = ResolutionAndComponentFlags();

   l = (CHECK_BIT(r, 3) == 8) ? 1 : 0; // in grib2 5th bit tells if uv is relative to grid or not (100 == 8 == true)
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
    SetLongKey("uvRelativeToGrid", static_cast<long> (theRelativity));
  }
  else
  {
    assert(itsHandle);
    long r = ResolutionAndComponentFlags();

  // http://stackoverflow.com/questions/47981/how-do-you-set-clear-and-toggle-a-single-bit-in-c-c

    if (theRelativity)
    {
      r |= 1 << 3;
  }
  else
  {
      r &= ~(1 << 3);
  }
    
  ResolutionAndComponentFlags(r);
  }
}

long NFmiGribMessage::ResolutionAndComponentFlags() const
{
  return GetLongKey("resolutionAndComponentFlags");
}

void NFmiGribMessage::ResolutionAndComponentFlags(long theResolutionAndComponentFlags)
{
  SetLongKey("resolutionAndComponentFlags", theResolutionAndComponentFlags);
}

void NFmiGribMessage::PackingType(const std::string& thePackingType)
{
  // Should probably check edition -- v2 has more packing types
  assert(itsHandle);
  size_t len = thePackingType.length();
  GRIB_CHECK(grib_set_string(itsHandle, "packingType", thePackingType.c_str(), &len), 0);

}

std::string NFmiGribMessage::PackingType() const
{
  return GetStringKey("packingType");
}

long NFmiGribMessage::TypeOfGeneratingProcess() const
{
  if (Edition() == 1)
  {
    return INVALID_INT_VALUE;
  }
  return GetLongKey("typeOfGeneratingProcess");
}

void NFmiGribMessage::TypeOfGeneratingProcess(long theProcess)
{
  SetLongKey("typeOfGeneratingProcess",theProcess);
}

void NFmiGribMessage::XLengthInMeters(double theLength) {
  SetDoubleKey("DxInMetres",theLength);
}

void NFmiGribMessage::YLengthInMeters(double theLength) {
  SetDoubleKey("DyInMetres",theLength);
}

double NFmiGribMessage::XLengthInMeters() const {
  return GetDoubleKey("DxInMetres");
}

double NFmiGribMessage::YLengthInMeters() const {
  return GetDoubleKey("DyInMetres");
}

long NFmiGribMessage::NormalizedUnitOfTimeRange() const
{

  // http://www.nco.ncep.noaa.gov/pmb/docs/on388/table4.html
  // http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-4.shtml

  long l;

  if (Edition() == 1) {
    l = GetLongKey("unitOfTimeRange");
  }

  else {
    l = GetLongKey("indicatorOfUnitOfTimeRange");

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
  if (Edition() == 1)
    return GetLongKey("unitOfTimeRange");
  else
    return GetLongKey("indicatorOfUnitOfTimeRange");
}

void NFmiGribMessage::UnitOfTimeRange(long theUnit)
{
  if (Edition() == 1)
    SetLongKey("unitOfTimeRange",theUnit);
  else
    SetLongKey("indicatorOfUnitOfTimeRange",theUnit);
}

/*
long NFmiGribMessage::UnitForTimeRange() const
{
  long l = INVALID_INT_VALUE;

  if (Edition() == 2)
    l = return GetLongKey("indicatorOfUnitForTimeRange");
  
  return l;
}
void NFmiGribMessage::UnitForTimeRange(long theUnit)
{
  if (Edition() == 2)
    GRIB_CHECK(grib_set_long(itsHandle,"indicatorOfUnitForTimeRange",theUnit), 0);
}
*/

long NFmiGribMessage::LengthOfTimeRange() const
{
  long l = INVALID_INT_VALUE;

  if (Edition() == 2)
  {
    l = GetLongKey("lengthOfTimeRange");
  }

  return l;
}

void NFmiGribMessage::LengthOfTimeRange(long theLength)
{
  if (Edition() == 2)
  {
    SetLongKey("lengthOfTimeRange",theLength);
  }
}

long NFmiGribMessage::TimeRangeIndicator() const
{
  return GetLongKey("timeRangeIndicator");  
}

void NFmiGribMessage::TimeRangeIndicator(long theTimeRangeIndicator)
{
  SetLongKey("timeRangeIndicator",theTimeRangeIndicator);
}

long NFmiGribMessage::P1() const
{
  return GetLongKey("P1");
}

void NFmiGribMessage::P1(long theP1)
{
  if (Edition() == 1)
    SetLongKey("P1",theP1);
}

long NFmiGribMessage::P2() const
{
  return GetLongKey("P2");
}

void NFmiGribMessage::P2(long theP2)
{
  if (Edition() == 1)
    SetLongKey("P2", theP2);
}

long NFmiGribMessage::NV() const
{
  return GetLongKey("NV");  
}

void NFmiGribMessage::NV(long theNV)
{
  // For some reason writing NV on grib2 disables the key "pv" !
  if (Edition() == 1)
    SetLongKey("NV",theNV);
}

std::vector<double> NFmiGribMessage::PV(size_t theNumberOfCoordinates, size_t level)
{
  assert(itsHandle);
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
  SetDoubleKey("PVPresent",1);
  GRIB_CHECK(grib_set_double_array(itsHandle,"pv",&theAB[0],abLen),0);
}

bool NFmiGribMessage::Write(const std::string &theFileName, bool appendToFile) {
  // Assume we have required directory structure in place
  assert(itsHandle);
  std::string mode = "w";

  if (appendToFile)
    mode = "a";

  // write compressed output if file extension is .gz or .bz2
  //--------------------------------------------------------------------------------------------
  boost::filesystem::path p (theFileName);
  std::string ext = p.extension().string();

  enum class file_compression { none, gzip, bzip2 };
  file_compression ofs_compression;

  // determine compression type for out file
  if (ext == ".gz")
  {
    ofs_compression = file_compression::gzip;
  }
  else if (ext == ".bz2")
  {
    ofs_compression = file_compression::bzip2;
  }
  else
  {
    ofs_compression = file_compression::none;
  }

  if (ofs_compression == file_compression::gzip || ofs_compression == file_compression::bzip2)
  {
    const void* buffer;
    size_t bfr_size;
    std::ofstream ofs;

    GRIB_CHECK(grib_get_message(itsHandle,&buffer,&bfr_size), 0);

    // copy data to stringstream as source for filtering ofstream
    std::string str_bfr(static_cast<const char*>(buffer),bfr_size);
    std::stringstream outdata(str_bfr);

    ofs.open(theFileName.c_str(), std::ofstream::out);
    boost::iostreams::filtering_streambuf<boost::iostreams::input> out;
    switch (ofs_compression)
    {
      case file_compression::gzip:
        out.push(boost::iostreams::gzip_compressor());
        break;
      case file_compression::bzip2:
        out.push(boost::iostreams::bzip2_compressor());
        break;
      case file_compression::none:
        break;
    }
    out.push(outdata);
    boost::iostreams::copy(out, ofs);

    return true;
  }
  //--------------------------------------------------------------------------------------------

  // if no file compression use grib_api function
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
  return GetSizeTKey(key);
}

bool NFmiGribMessage::Bytes(const std::string& key, unsigned char* data) const
{
  assert(itsHandle);
  size_t length = BytesLength(key);
  
  GRIB_CHECK(grib_get_bytes(itsHandle,key.c_str(),data, &length),0);

  return true;
}

bool NFmiGribMessage::PackedValues(unsigned char* data) const
{
#ifdef GRIB_READ_PACKED_DATA
  assert(itsHandle);
  size_t dataLength;

  GRIB_CHECK(grib_get_packed_values(itsHandle,data,&dataLength),0);

  assert(dataLength == PackedValuesLength());

#else
#warning GRIB_READ_PACKED_DATA not defined -- reading packed data with fmigrib is not supported
  throw std::runtime_error("This version on NFmiGrib is not compiled with support for reading of packed data");
#endif

  return true;
}

bool NFmiGribMessage::PackedValues(unsigned char* data, size_t unpacked_len, int* bitmap, size_t bitmap_len)
{
#ifdef GRIB_WRITE_PACKED_DATA
  assert(itsHandle);

  GRIB_CHECK(grib_set_packed_values(itsHandle, data, unpacked_len, bitmap, bitmap_len), 0);
#else
//#warning GRIB_WRITE_PACKED_DATA not defined -- writing of packed data with fmigrib is not supported
  throw std::runtime_error("This version on NFmiGrib is not compiled with support for writing of packed data");
#endif

  return true;
}

long NFmiGribMessage::BinaryScaleFactor() const
{
  return GetLongKey("binaryScaleFactor");
}

void NFmiGribMessage::BinaryScaleFactor(long theFactor)
{
#ifdef GRIB_WRITE_PACKED_DATA
  assert(itsHandle);
  GRIB_CHECK(grib_set_long_internal(itsHandle, "binaryScaleFactor", theFactor), 0);
#endif
}


long NFmiGribMessage::DecimalScaleFactor() const
{
  return GetLongKey("decimalScaleFactor");
}

void NFmiGribMessage::DecimalScaleFactor(long theFactor)
{
#ifdef GRIB_WRITE_PACKED_DATA
  assert(itsHandle);
  GRIB_CHECK(grib_set_long_internal(itsHandle, "decimalScaleFactor", theFactor), 0);
#endif
}

double NFmiGribMessage::ReferenceValue() const
{
  return GetDoubleKey("referenceValue");
}

void NFmiGribMessage::ReferenceValue(double theValue)
{
#ifdef GRIB_WRITE_PACKED_DATA
  assert(itsHandle);
  GRIB_CHECK(grib_set_double_internal(itsHandle,"referenceValue",theValue), 0);
#endif
}

long NFmiGribMessage::Section4Length() const
{
  return GetLongKey("section4Length");
}

bool NFmiGribMessage::KeyExists(const std::string& theKey) const
{
  int i;
  assert(itsHandle);
  
  i = grib_is_defined(itsHandle, theKey.c_str());
  return (i == 0 ? false: true);
}

long NFmiGribMessage::GetLongKey(const std::string& keyName) const
{
  assert(itsHandle);
  long l;
  
  int err = grib_get_long(itsHandle,keyName.c_str(),&l);

  if (err != 0)
  {
    l = INVALID_INT_VALUE;
  }
  
  return l;
}

void NFmiGribMessage::SetLongKey(const std::string& keyName, long value)
{
  assert(itsHandle);
  
  int err = grib_set_long(itsHandle,keyName.c_str(),value);
  
  if (err != 0)
  {
    throw err;
  }
}

double NFmiGribMessage::GetDoubleKey(const std::string& keyName) const
{
  double d;
  
  assert(itsHandle);
  
  int err = grib_get_double(itsHandle,keyName.c_str(),&d);
  
  if (err != 0)
  {
    d = static_cast<double> (INVALID_INT_VALUE);
  }

  return d;
}

void NFmiGribMessage::SetDoubleKey(const std::string& keyName, double value)
{
  assert(itsHandle);
  
  GRIB_CHECK(grib_set_double(itsHandle,keyName.c_str(),value), 0);
}

size_t NFmiGribMessage::GetSizeTKey(const std::string& keyName) const
{
  size_t s;
  
  assert(itsHandle);
  
  GRIB_CHECK(grib_get_size(itsHandle,keyName.c_str(),&s), 0);

  return s;
}

std::string NFmiGribMessage::GetStringKey(const std::string& keyName) const
{
  size_t len = 1024;
  char s[len];
  
  assert(itsHandle);

  GRIB_CHECK(grib_get_string(itsHandle,keyName.c_str(), s, &len), 0);

  return std::string(s);
}

long NFmiGribMessage::Type() const
{
  // http://old.ecmwf.int/publications/manuals/d/gribapi/mars/att=type/  
  return GetLongKey("type");
}

long NFmiGribMessage::ForecastType() const
{
  long forecastType = 1; // deterministic

  if (Edition() == 1)
  {
    if (KeyExists("localDefinitionNumber"))
    {
    long definitionNumber = GetLongKey("localDefinitionNumber");
    // EC uses local definition number in Grib1
      // http://old.ecmwf.int/publications/manuals/d/gribapi/fm92/grib1/show/local/
  
      switch (definitionNumber)
      {
        case 0:
          // no local definition --> deterministic
          forecastType = 1;
          break;

        case 1:
          // MARS labeling or ensemble forecast data
        {
          long definitionType = Type();

          switch (definitionType)
          {
            case 9:
              // deterministic forecast
              forecastType = 1;
              break;    
          
          case 10:
              // cf -- control forecast
              forecastType = 4;
              break;
            case 11:
              // pf -- perturbed forecast
              forecastType = 3;
              break;
            default:
              break;
          }
        break;
        }
        default:
          break;
      }
    }
  }
  else
  {
    long typeOfGeneratingProcess = TypeOfGeneratingProcess();

    switch (typeOfGeneratingProcess)
    {
      case 0:
        // Analysis
       forecastType = 2;
       break;

      case 2:
        // deterministic
        forecastType = 1;
        break;

      case 4:
        // eps
      {
        long typeOfEnsemble = GetLongKey("typeOfEnsembleForecast");

        switch (typeOfEnsemble)
        {
          case 0:
          case 1:
            // control forecast
            forecastType = 4;
            break;

          case 2:
          case 3:
          case 192:
            // perturbed forecast
            forecastType = 3;
            break;

          default:
            break;
        }
        break;
      }

    default:
      break;
    }  
  }
  
  return forecastType;
}

void NFmiGribMessage::ForecastType(long theForecastType)
{
  // todo
}
  
double NFmiGribMessage::ForecastTypeValue() const
{
  long forecastType = ForecastType();
  
  if (forecastType == 1 || forecastType == 2 || forecastType == 4)
  {
    return static_cast<double> (INVALID_INT_VALUE);
  }
  
  assert(forecastType == 3);
  
  return static_cast<double> (PerturbationNumber());
  
}

void NFmiGribMessage::ForecastTypeValue(double theForecastTypeValue)
{
  // todo
}

#ifdef GRIB_WRITE_PACKED_DATA
double NFmiGribMessage::CalculateReferenceValue(double minimumValue)
{
  assert(itsHandle);
  double ref = 0;
  GRIB_CHECK(grib_get_reference_value(itsHandle, minimumValue, &ref), 0);
  return ref;
}
#endif

#if defined HAVE_CUDA && defined GRIB_READ_PACKED_DATA

bool NFmiGribMessage::CudaUnpack(double* arr, size_t len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CudaUnpack(arr, len, stream);
  CUDA_CHECK(cudaStreamDestroy(stream)); // synchronizes obviously
  return true;
}

bool NFmiGribMessage::CudaUnpack(double* arr, size_t unpackedLen, cudaStream_t& stream)
{
  using namespace NFmiGribPacking;

  assert(unpackedLen == ValuesLength());
  assert(itsHandle);
  assert(arr);
 
  // 1. Get packed values from grib

  unsigned char* packed = 0;
  size_t packedLen = PackedValuesLength();
  CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&packed), packedLen * sizeof(unsigned char)));
  PackedValues(packed);

  // 2. If bitmap is present, unpack it and copy to cuda device
  
  int* d_bitmap = 0;

  if (Bitmap())
  {
    size_t bitmap_len = BytesLength("bitmap");
    assert(bitmap_len == unpackedLen);
    
    size_t bitmap_size = static_cast<size_t> (ceil(static_cast<double> (bitmap_len)/8));
    //CUDA_CHECK(cudaMemcpyAsync(d_bitmap, bitmap, bitmapLength * sizeof(int), cudaMemcpyHostToDevice, *stream));

    unsigned char* bitmap = new unsigned char[bitmap_size];

    Bytes("bitmap", bitmap);

    int* unpacked_bitmap = 0;
    
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&unpacked_bitmap), bitmap_len * sizeof(int)));

    UnpackBitmap(bitmap, unpacked_bitmap, bitmap_size, bitmap_len);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**> (&d_bitmap), bitmap_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_bitmap, unpacked_bitmap, bitmap_len * sizeof(int), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // 3. Set up coefficients

  packing_coefficients coeffs;
  coeffs.bitsPerValue = static_cast<int> (BitsPerValue());
  coeffs.binaryScaleFactor = ToPower(static_cast<double>(BinaryScaleFactor()),2);
  coeffs.decimalScaleFactor = ToPower(-static_cast<double>(DecimalScaleFactor()), 10);
  coeffs.referenceValue = ReferenceValue();
  
#ifdef DEBUG
  std::cout << "grib packing type: " << PackingType() << std::endl;
#endif

  if (PackingType() == "grid_simple")
  {
    simple_packing::Unpack(arr, packed, d_bitmap, unpackedLen, coeffs, stream);
  }
  else if (PackingType() == "grid_jpeg")
  {
    jpeg_packing::Unpack(arr, packed, d_bitmap, unpackedLen, coeffs, stream);
  }
  else {
    CUDA_CHECK(cudaFreeHost(packed));
    return false;
  }

  CUDA_CHECK(cudaFreeHost(packed));
  
  return true;

}

bool NFmiGribMessage::CudaPack(double* arr, size_t unpackedLen)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CudaPack(arr, unpackedLen, stream);
  CUDA_CHECK(cudaStreamDestroy(stream)); // synchronizes obviously
  return true;
}

bool NFmiGribMessage::CudaPack(double* arr, size_t unpackedLen, cudaStream_t& stream)
{
  using namespace NFmiGribPacking;
  assert(itsHandle);

  // 1. No bitmap support for now

  assert(!Bitmap());

  // 3. Set up coefficients

  packing_coefficients coeffs;

  coeffs.bitsPerValue = static_cast<int> (BitsPerValue());

  double min, max;
  MinMax(arr, unpackedLen, min, max, stream);

  coeffs.referenceValue = CalculateReferenceValue(min);

  // TODO: voidaanko tehdä grib_apissa ja vain paljastaa funktiokutsu fmigribille?

  coeffs.binaryScaleFactor = static_cast<double> (simple_packing::get_binary_scale_fact(max, min, static_cast<long> (coeffs.bitsPerValue)));
  coeffs.decimalScaleFactor = static_cast<double> (simple_packing::get_decimal_scale_fact(max, min, static_cast<long> (coeffs.bitsPerValue), static_cast<long> (coeffs.binaryScaleFactor)));

#if 0  
  std::cout << "bits per value: " << coeffs.bitsPerValue << std::endl
            << "binary scale factor: " << coeffs.binaryScaleFactor << std::endl
            << "decimal scale factor: " << coeffs.decimalScaleFactor << std::endl
            << "reference value: " << coeffs.referenceValue << std::endl;
#endif
            
  long packedLen = ((BitsPerValue()*unpackedLen)+7)/8;

  unsigned char* packed = 0;
  CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&packed), packedLen * sizeof(unsigned char)));
  
#ifdef DEBUG
  std::cout << "grib packing type: " << PackingType() << std::endl;
#endif

  if (PackingType() == "grid_simple")
  {
    simple_packing::Pack(arr, packed, 0, unpackedLen, coeffs, stream);
  }
  else if (PackingType() == "grid_jpeg")
  {
    jpeg_packing::Pack(arr, packed, 0, unpackedLen, coeffs, stream);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  PackedValues(packed, unpackedLen, 0, 0);

  CUDA_CHECK(cudaFreeHost(packed));
  
  return true;
}

#else
bool NFmiGribMessage::CudaUnpack(double* arr, size_t len)
{
#ifndef HAVE_CUDA
  std::cerr << "CUDA support disabled at compile time" << std::endl;
  return false;
#else
#ifndef GRIB_READ_PACKED_DATA
  std::cerr << "grib_api does not support reading of packed data" << std::endl;
  return false;
#endif
#endif
}

bool NFmiGribMessage::CudaPack(double* arr, size_t len)
{
#ifndef HAVE_CUDA
  std::cerr << "CUDA support disabled at compile time" << std::endl;
  return false;
#else
#ifndef GRIB_WRITE_PACKED_DATA
  std::cerr << "grib_api does not support writing of packed data" << std::endl;
  return false;
#endif
#endif

}

#endif
