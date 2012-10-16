/*
 * NFmiGribMessage.h
 *
 *  Created on: Oct 16, 2012
 *      Author: partio
 */

#ifndef NFMIGRIBMESSAGE_H_
#define NFMIGRIBMESSAGE_H_

#include <grib_api.h>
#include <string>

class NFmiGribMessage {

  public:

	NFmiGribMessage() { Clear(); }
	~NFmiGribMessage() {}

	bool Read(grib_handle *h);

    long SizeX();
    long SizeY();
    long SizeZ();

    double X0();
    double Y0();

    double SouthPoleX();
    double SouthPoleY();

    double *Values();
    int ValuesLength();

    long DataDate();
    long DataTime();
    long ForecastTime();

    long ParameterNumber();
    long ParameterDiscipline();
    long ParameterCategory();

    std::string ParameterName();

    long GridType();

    double XResolution();
    double YResolution();

    long Edition();

    double GridOrientation();

    long Centre() { return itsCentre; }

    long Year() { return itsYear; }
    long Month() { return itsMonth; }
    long Day() { return itsDay; }
    long Hour() { return itsHour; }
    long Minute() { return itsMinute; }

    long Process() { return itsProcess; }
    long Table2Version() { return itsTable2Version; }
    long LevelType();
    long Level() { return itsLevel; }

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
    long EndStep() { return itsEndStep; }
    long StepUnits() { return itsStepUnits; }
    long StepRange() { return itsStepRange; }

    long TimeRangeIndicator() { return itsTimeRangeIndicator; }

  private:
    void Clear();

    double *itsValues;

    long itsEdition;
    long itsProcess;
    long itsCentre;

    long itsXSize;
    long itsYSize;

    double itsLatitudeOfFirstGridPoint;
    double itsLongitudeOfFirstGridPoint;
    double itsLatitudeOfLastGridPoint;
    double itsLongitudeOfLastGridPoint;

    long itsIScansNegatively;
    long itsJScansPositively;

    double itsLatitudeOfSouthernPole;
    double itsLongitudeOfSouthernPole;

    long itsIndicatorOfParameter;
    long itsIndicatorOfTypeOfLevel;

    long itsLevel;
    long itsDate;
    long itsTime;
    long itsForecastTime;

    long itsStepUnits;
    long itsStepRange;
    long itsStartStep;
    long itsEndStep;

    long itsGridType;
    long itsGridDefinitionTemplate;

    size_t itsValuesLength;

    long itsParameterDiscipline;
    long itsParameterCategory;
    long itsParameterNumber;

    std::string itsParameterName;
    long itsTypeOfFirstFixedSurface;

    double itsXResolution;
    double itsYResolution;

    double itsOrientationOfTheGrid;

    long itsBitmapPresent;

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

};


#endif /* NFMIGRIBMESSAGE_H_ */
