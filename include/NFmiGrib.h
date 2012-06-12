/*
 * class NFmiGrib
 *
 * C++ overcout for reading grib data and metadata.
 */

#ifndef __NFMIGRIB_H__
#define __NFMIGRIB_H__

#include <string>
#include <grib_api.h>

class NFmiGrib {

  public:
    NFmiGrib();
    NFmiGrib(const std::string &theFileName);
    ~NFmiGrib();

    bool Open(const std::string &theFileName);
    bool Read();

    long SizeX();
    long SizeY();
    long SizeZ();

    double X0();
    double Y0();

    double SouthPoleX();
    double SouthPoleY();

    bool NextMessage();
    int MessageCount();
    int CurrentMessageIndex();

    double *Data();
    int DataLength();

    long DataDate();
    long DataTime();

    long ParameterNumber();
    long ParameterDiscipline();
    long ParameterCategory();

    long GridType();
    long NormalizedGridType();

    double XResolution();
    double YResolution();

    long Edition();

  private:

    void Clear();

    grib_handle *h;
    FILE *f;

    double *values;

    long itsEdition;
    long itsProcess;
    long itsCentre;
    int itsMessageCount;
    int itsCurrentMessage;

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

    long itsStepUnits;
    long itsStepRange;
    long itsStartStep;
    long itsEndStep;

    long itsGridType;
    long itsGridDefinitionTemplate;

    size_t itsDataLength;

    long itsParameterDiscipline;
    long itsParameterCategory;
    long itsParameterNumber;
    long itsTypeOfFirstFixedSurface;

    double itsXResolution;
    double itsYResolution;

}; 

#endif

