/*
 * class NFmiGrib
 *
 * C++ overcoat for reading grib data and metadata.
 */

#ifndef __NFMIGRIB_H__
#define __NFMIGRIB_H__

#include <string>
#include <grib_api.h>
#include "NFmiGribMessage.h"
#include <memory>

const int INVALID_INT_VALUE = -999;

class NFmiGrib {

  public:
    NFmiGrib();
    ~NFmiGrib();

    bool Open(const std::string &theFileName);
    bool Read();

    bool NextMessage();
    int MessageCount();
    int CurrentMessageIndex();

    void MultiGribSupport(bool theMultiGribSupport);
    bool WriteMessage(const std::string &theFileName);

    NFmiGribMessage& Message() { return m; }

  private:

    grib_handle *h;
    FILE *f;

    int itsMessageCount;
    int itsCurrentMessage;

    NFmiGribMessage m;

}; 

#endif

