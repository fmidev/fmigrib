#include "NFmiGrib.h"
#include <stdexcept>
#include <iostream>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/stream.hpp>

const float kFloatMissing = 32700;

NFmiGrib::NFmiGrib() :
  ifs_compression(file_compression::none),
  ofs_compression(file_compression::none),
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

/*
 * Check if input file is gzip packed
 */

//-----------------------------------------------------------
  if (theFileName.rfind("grib.gz") != std::string::npos)
  {
    // set packed mode
    ifs_compression = file_compression::gzip;
    
    // Open input file into input stream
    ifs.open(theFileName.c_str(), std::ifstream::binary);
    return true;
  }
//-----------------------------------------------------------

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

/* 
 * Read data through filtering streambuf if gzip packed.
 */

//------------------------------------------------------------------------------
  if (ifs_compression == file_compression::gzip || ifs_compression == file_compression::bzip)
  {
    //stringstream serves as sink for the input filter
    std::stringstream str_buffer;
    
    //create input filter
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::gzip_decompressor());
    in.push(ifs);

    //copy data to stringbuffer
    size_t size = boost::iostreams::copy(in,str_buffer);

    //create grib_message from stringbuffer
    if ((h = grib_handle_new_from_message_copy(0,str_buffer.str().c_str(),size*sizeof(char))) != NULL) {
      itsCurrentMessage++;
      assert(h);

      return m.Read(h);
    }
    
    return false;
  }
//-----------------------------------------------------------------------------

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
