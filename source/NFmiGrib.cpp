#include "NFmiGrib.h"
#include <stdexcept>
#include <iostream>

const int INVALID_INT_VALUE = -999;
const float kFloatMissing = 32700;

NFmiGrib::NFmiGrib() :
  h(0),
  f(0),
  itsMessageCount(0),
  itsCurrentMessage(0)
{
  m = std::shared_ptr<NFmiGribMessage> (new NFmiGribMessage());
}

NFmiGrib::NFmiGrib(const std::string &theFileName) :
  h(0),
  f(0),
  itsMessageCount(0),
  itsCurrentMessage(0)
{

  m = std::shared_ptr<NFmiGribMessage> (new NFmiGribMessage());
  Open(theFileName);
}

NFmiGrib::~NFmiGrib() {

  if (h)
    grib_handle_delete(h);

  if (f)
    fclose(f);

}

bool NFmiGrib::Open(const std::string &theFileName) {

  if (f)
    fclose(f);

  // Open with 'rb', although in linux it equals to 'r'

  if (!(f = fopen(theFileName.c_str(), "rb")))
    return false;

  if (grib_count_in_file(0, f, &itsMessageCount) != GRIB_SUCCESS)
    return false;

  return true;
}

bool NFmiGrib::NextMessage() {

  int err;

  if (h)
    grib_handle_delete(h);
  
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
    grib_multi_support_on(0);
  else
    grib_multi_support_off(0);
}

bool NFmiGrib::WriteMessage(const std::string &theFileName) {
  // Assume we have required directory structure in place

  GRIB_CHECK(grib_write_message(h, theFileName.c_str(), "w"), 0);

  return true;
}
