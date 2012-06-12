/*
 * class NFmiGrib
 *
 * C++ overcout for reading grib data and metadata.
 */

#include <string>
#include <grib_api.h>

class NFmiGrib {

  public:
    NFmiGrib();
    ~NFmiGrib();

    bool Open(const std::string &theFileName);
    bool Read();

    int SizeX();
    int SizeY();

  private:
    grib_handle *h;
    FILE *f;

    unsigned short itsEdition;

    int itsXSize;
    int itsYSize;
}; 
