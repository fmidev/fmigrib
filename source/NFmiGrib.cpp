#include "NFmiGrib.h"
#include <stdexcept>

NFmiGrib::NFmiGrib() {}
NFmiGrib::~NFmiGrib() {

  if (h)
    grib_handle_delete(h);

  if (f)
    fclose(f);
}

bool NFmiGrib::Open(const std::string &theFileName) {

  int err = 0;

  // Open with 'rb', although in linux it equals to 'r'

  if (f)
    fclose(f);

  if (!(f = fopen(theFileName.c_str(), "rb")))
    return false;

  if (h) 
    grib_handle_delete(h);

  h = grib_handle_new_from_file(0, f, &err);
  
  if(err != GRIB_SUCCESS) {
    return false;
  }

  GRIB_CHECK(grib_set_long(h,"edition",itsEdition),0);

  return Read();
}

int NFmiGrib::SizeX() {
  return itsXSize;

}

bool NFmiGrib::Read() {

  if (!f)
    return false;

  if (!f)
    return false;

  if (itsEdition == 1) {

  }
  else if (itsEdition == 2) {


  }
  else 
    throw std::runtime_error("Unknown grib edition");

  return true;
}




