/*
 * NFmiGribMessage.h
 *
 *  Created on: Oct 16, 2012
 *	  Author: partio
 *
 * One NFmiGribMessage equals to one grib message.
 * All the setter functions modify in-memory structures,
 * the changes are are materialized only when the message
 * is written to disk with WriteMessage().
 */

#ifndef NFMIGRIBMESSAGE_H_
#define NFMIGRIBMESSAGE_H_

#include <grib_api.h>
#include <string>
#include <boost/bimap.hpp>

class NFmiGribMessage {

  public:

	NFmiGribMessage();
	~NFmiGribMessage();

	bool Read(grib_handle *h);

	long SizeX() const;
	long SizeY() const;
	//long SizeZ();

	void SizeX(long theXSize);
	void SizeY(long theYSize);
	//void SizeZ();

	double X0() const;
	double Y0() const;

	void X0(double theX0);
	void Y0(double theY0);

	double X1() const;
	double Y1() const;

	void X1(double theX1);
	void Y1(double theY1);

	double SouthPoleX() const;
	double SouthPoleY() const;

	double *Values() ;
	void Values(const double* theValues, long theValuesLength);

	int ValuesLength() const;

	long DataDate() const;
	void DataDate(long theDate);

	long DataTime() const;
	void DataTime(long theTime);

	long ForecastTime() const;
	void ForecastTime(long theTime);

	std::string ParameterUnit() const;
	long ParameterNumber() const;
	long ParameterDiscipline() const;
	long ParameterCategory() const;

	void ParameterNumber(long theNumber);
	void ParameterDiscipline(long theDiscipline);
	void ParameterCategory(long theCategory);

	std::string ParameterName() const;

	long GridType() const;
	void GridType(long theGridType);

	double XResolution() const;
	double YResolution() const;

	long Edition() const;
	void Edition(long theEdition);

	double GridOrientation() const;
	void GridOrientation(double theGridOrientation);

	long Centre() const;
	void Centre(long theCentre);

	long Year() { return itsYear; }
	long Month() { return itsMonth; }
	long Day() { return itsDay; }
	long Hour() { return itsHour; }
	long Minute() { return itsMinute; }

	long Process() const;
	void Process(long theProcess);

	long Table2Version() const;
	void Table2Version(long theVersion);

	long DataType() { return itsDataType; }
	long PerturbationNumber() { return itsPerturbationNumber; }

	long NormalizedGridType(unsigned int targetEdition = 1) const;
	long NormalizedLevelType(unsigned int targetEdition = 1) const;

	long LocalDefinitionNumber() { return itsLocalDefinitionNumber; }
	long DerivedForecast() { return itsDerivedForecast; }
	long TypeOfEnsembleForecast() { return itsTypeOfEnsembleForecast; }
	long NumberOfForecastsInTheEnsemble() { return itsNumberOfForecastsInTheEnsemble; }
	long ClusterIdentifier() { return itsClusterIdentifier; }

	long ForecastProbabilityNumber() { return itsForecastProbabilityNumber; }
	long ProbabilityType() { return itsProbabilityType; }
	long PercentileValue() { return itsPercentileValue; }
	long NumberOfTimeRange() { return itsNumberOfTimeRange; }
	long TypeOfTimeIncrement() { return itsTypeOfTimeIncrement; }

	long StartStep() const;
	void StartStep(long theStartStep);

	long EndStep() const;
	void EndStep(long theEndStep);

	long StepUnits() const;
	void StepUnits(long theUnits);

	long StepRange() const;
	void StepRange(long theRange);

	long TimeRangeIndicator() const { return itsTimeRangeIndicator; }

	bool Write(const std::string& theOutputFile);

	void Year(const std::string& theYear);
	void Month(const std::string& theMonth);
	void Day(const std::string& theDay);
	void Hour(const std::string& theHour);
	void Minute(const std::string& theMinute);
	void Second(const std::string& theSecond);

	bool Bitmap() const;
	void Bitmap(bool theBitmap);

	void PackingType(const std::string& thePackingType);
	std::string PackingType() const;

	long LevelValue() const;
	void LevelValue(long theLevelValue);

	long LevelType() const;
	void LevelType(long theLevelType);

	bool IScansNegatively() const;
	bool JScansPositively() const;

	double iDirectionIncrement() const;
	void iDirectionIncrement(double theIncrement);

	double jDirectionIncrement() const;
	void jDirectionIncrement(double theIncrement);

	long TypeOfGeneratingProcess() const;
	void TypeOfGeneratingProcess(long theProcess);

	// Are these valid ?
	void XLengthInMeters(double theLength);
	void YLengthInMeters(double theLength);

	long GridTypeToAnotherEdition(long gridType, long edition) const;
	long LevelTypeToAnotherEdition(long levelType, long edition) const;

	long NumberOfMissing() const;

  private:
	void Clear();

	size_t itsValuesLength;

	long itsTotalLength;

	long itsYear;
	long itsMonth;
	long itsDay;
	long itsHour;
	long itsMinute;

	long itsTimeRangeIndicator;
	long itsLocalDefinitionNumber;

	long itsPerturbationNumber;
	long itsDataType;
	long itsTypeOfEnsembleForecast;
	long itsDerivedForecast;
	long itsNumberOfForecastsInTheEnsemble;
	long itsClusterIdentifier;
	long itsForecastProbabilityNumber;
	long itsProbabilityType;
	long itsPercentileValue;
	long itsNumberOfTimeRange;
	long itsTypeOfTimeIncrement;

	grib_handle *itsHandle;

	boost::bimap<long,long> itsGridTypeMap;
	boost::bimap<long,long> itsLevelTypeMap;
};

#endif /* NFMIGRIBMESSAGE_H_ */
