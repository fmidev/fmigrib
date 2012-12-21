/*
 * NFmiGribMessage.h
 *
 *  Created on: Oct 16, 2012
 *      Author: partio
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

class NFmiGribMessage {

  public:

	NFmiGribMessage();
	~NFmiGribMessage();

	bool Read(grib_handle *h);

    long SizeX();
    long SizeY();
    //long SizeZ();

    void SizeX(long theXSize);
    void SizeY(long theYSize);
    //void SizeZ();

    double X0();
    double Y0();

    void X0(double theX0);
    void Y0(double theY0);

    double X1();
    double Y1();

    void X1(double theX1);
    void Y1(double theY1);

    double SouthPoleX();
    double SouthPoleY();

    double *Values();
    void Values(const double* theValues, long theValuesLength);

    int ValuesLength();

    long DataDate();
    long DataTime();
    long ForecastTime();

    long ParameterNumber();
    long ParameterDiscipline();
    long ParameterCategory();

    void ParameterNumber(long theNumber);
    void ParameterDiscipline(long theDiscipline);
    void ParameterCategory(long theCategory);

    std::string ParameterName();

    long GridType();
    void GridType(long theGridType);

    double XResolution();
    double YResolution();

    long Edition();
    void Edition(long theEdition);

    double GridOrientation();
    void GridOrientation(double theGridOrientation);

    long Centre();
    void Centre(long theCentre);

    long Year() { return itsYear; }
    long Month() { return itsMonth; }
    long Day() { return itsDay; }
    long Hour() { return itsHour; }
    long Minute() { return itsMinute; }

    long Process();
    void Process(long theProcess);

    long Table2Version() { return itsTable2Version; }
    long LevelType();
    //long Level() { return itsLevel; }

    long DataType() { return itsDataType; }
    long PerturbationNumber() { return itsPerturbationNumber; }

    long NormalizedGridType();
    long NormalizedLevelType();

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

    long StartStep() { return itsStartStep; }
    void StartStep(long theStartStep);

    long EndStep() { return itsEndStep; }
    void EndStep(long theEndStep);

    long StepUnits() { return itsStepUnits; }
    long StepRange() { return itsStepRange; }

    long TimeRangeIndicator() { return itsTimeRangeIndicator; }

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

    // Are these valid ?
    void XLengthInMeters(double theLength);
    void YLengthInMeters(double theLength);

  private:
    void Clear();

    double *itsValues;

    long itsIScansNegatively;
    long itsJScansPositively;

    long itsIndicatorOfTypeOfLevel;

    long itsDate;
    long itsTime;
    long itsForecastTime;

    long itsStepUnits;
    long itsStepRange;
    long itsStartStep;
    long itsEndStep;

    size_t itsValuesLength;

    long itsTypeOfFirstFixedSurface;

    double itsXResolution;
    double itsYResolution;

    long itsTotalLength;
    long itsTable2Version;

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
};

#endif /* NFMIGRIBMESSAGE_H_ */
