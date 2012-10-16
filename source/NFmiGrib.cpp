#include "NFmiGrib.h"
#include <stdexcept>
#include <iostream>

const int INVALID_INT_VALUE = -999;
const float kFloatMissing = 32700;

NFmiGrib::NFmiGrib() :
  h(0),
  f(0),
  itsMessageCount(0),
  itsCurrentMessage(0),
  m(0)
{
  m = new NFmiGribMessage;
}

NFmiGrib::NFmiGrib(const std::string &theFileName) :
  h(0),
  f(0),
  itsMessageCount(0),
  itsCurrentMessage(0),
  m(0)
  {

  m = new NFmiGribMessage;
  Open(theFileName);
}

NFmiGrib::~NFmiGrib() {

  if (h)
    grib_handle_delete(h);

  if (f)
    fclose(f);

  if (m)
    delete m;

}

bool NFmiGrib::Open(const std::string &theFileName) {

   // Open with 'rb', although in linux it equals to 'r'

  if (f)
    fclose(f);

  if (!(f = fopen(theFileName.c_str(), "rb")))
    return false;

  if (grib_count_in_file(0, f, &itsMessageCount) != GRIB_SUCCESS)
    return false;

  //if (h)
  //  grib_handle_delete(h);

  grib_multi_support_on(0); // Multigrib support on

  return true;
}

bool NFmiGrib::NextMessage() {

  int err;

  if ((h = grib_handle_new_from_file(0,f,&err)) != NULL) {
    itsCurrentMessage++;

    Message()->Read(h);
    return true;

  }
  else
    return false;
}

int NFmiGrib::MessageCount() {
  return itsMessageCount;
}

int NFmiGrib::CurrentMessageIndex() {
  return itsCurrentMessage;
}

void NFmiGrib::MultiGribSupport(bool theMultiGribSupport) {
  if (theMultiGribSupport)
    grib_multi_support_on(0); // Multigrib support on
  else
    grib_multi_support_off(0); // Multigrib support on
}

bool NFmiGrib::WriteMessage(const std::string &theFileName) {
  // Assume we have required directory structure in place

  GRIB_CHECK(grib_write_message(h, theFileName.c_str(), "w"), 0);

  return true;
}
