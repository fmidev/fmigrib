#include "NFmiGrib.h"
#include <stdexcept>
#include <iostream>

const float kFloatMissing = 32700;

NFmiGrib::NFmiGrib() :
  h(0),
  f(0),
  itsMessageCount(INVALID_INT_VALUE),
  itsCurrentMessage(0)
{
  h = grib_handle_new_from_samples(NULL,"GRIB2");
  assert(h);
  m.Read(h);
}

NFmiGrib::~NFmiGrib() {

  if (h)
  {
    grib_handle_delete(h);
  }
  if (f)
  {
    fclose(f);
  }
}

bool NFmiGrib::Open(const std::string &theFileName) {

  if (f)
  {
    fclose(f);
	f = 0;
  }
    
  // Open with 'rb', although in linux it equals to 'r'

  if (!(f = fopen(theFileName.c_str(), "rb")))
  {
    return false;
  }
  
  return true;
}

bool NFmiGrib::NextMessage() {

  int err;

  if (h) {
    // this invalidates itaHandle @ m
    grib_handle_delete(h);
    h = 0;
  }
  
  assert(!h);
  
  if ((h = grib_handle_new_from_file(0,f,&err)) != NULL) {
    itsCurrentMessage++;
    assert(h);

    return m.Read(h);
  }
  
  return false;
 
}

int NFmiGrib::MessageCount() {

  if (itsMessageCount == INVALID_INT_VALUE)
  {
    GRIB_CHECK(grib_count_in_file(0, f, &itsMessageCount), 0);
  }

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
