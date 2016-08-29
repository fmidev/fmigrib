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
#include <vector>
#include <fstream>

const int INVALID_INT_VALUE = -999;

class NFmiGrib {

  public:
    NFmiGrib();
    ~NFmiGrib();

    bool Open(const std::string &theFileName);
    bool BuildIndex(const std::string &theFileName, const std::vector<std::string> &theKeys);
    bool BuildIndex(const std::string &theFileName, const std::string &theKeys);

    bool AddFileToIndex(const std::string &theFileName);

    std::vector<long> GetIndexValues(const std::string &theKey);

    bool Message(const std::map<std::string, long> &theKeyValue);
    bool NextMessage();
    int MessageCount();
    int CurrentMessageIndex();

    void MultiGribSupport(bool theMultiGribSupport);
    bool WriteMessage(const std::string &theFileName);
    bool WriteIndex(const std::string &theFileName);

    NFmiGribMessage& Message() { return m; }

  private:

    enum class file_compression { none, gzip, bzip2 };
    file_compression ifs_compression;
    file_compression ofs_compression;
    std::ifstream ifs;
    std::ofstream ofs;

    //string serves as sink for the input filter
    std::string ifile;
    size_t message_start;
    size_t message_end;

    grib_handle *h;
    grib_index  *index;
    FILE *f;

    int itsMessageCount;
    int itsCurrentMessage;

    NFmiGribMessage m;

}; 

#endif

