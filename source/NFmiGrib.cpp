/*
 * NFmiGrib.cpp
 *
*/

#include "NFmiGrib.h"
#include <iostream>
#include <stdexcept>

#include <boost/filesystem/path.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/stream.hpp>

const float kFloatMissing = 32700;

NFmiGrib::NFmiGrib()
    : ifs_compression(file_compression::none),
      ofs_compression(file_compression::none),
      message_start(0),
      message_end(0),
      h(0),
      index(0),
      f(0),
      itsMessageCount(INVALID_INT_VALUE),
      itsCurrentMessage(0)
{
	h = grib_handle_new_from_samples(NULL, "GRIB2");
	assert(h);
	m.Read(h);
}

NFmiGrib::~NFmiGrib()
{
	if (h)
	{
		grib_handle_delete(h);
	}
	if (index)
	{
		grib_index_delete(index);
	}
	if (f)
	{
		fclose(f);
	}
	if (ifs.is_open())
	{
		ifs.close();
	}
	if (ofs.is_open())
	{
		ofs.close();
	}
}

bool NFmiGrib::Open(const std::string &theFileName)
{
	/*
	 * Check if input file is gzip packed
	 */

	if (ifs.is_open())
	{
		ifs.close();
	}

	boost::filesystem::path p(theFileName);

	std::string ext = p.extension().string();

	if (ext == ".idx")
	{
		int ret = 0;
		index = grib_index_read(0, theFileName.c_str(), &ret);
		if (ret == 0)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	if (ext == ".gz")
	{
		// set packed mode
		ifs_compression = file_compression::gzip;
	}
	else if (ext == ".bz2")
	{
		ifs_compression = file_compression::bzip2;
	}
	else
	{
		ifs_compression = file_compression::none;
	}

	if (ifs_compression == file_compression::gzip || ifs_compression == file_compression::bzip2)
	{
		// Open input file into input stream
		ifs.open(theFileName.c_str(), std::ifstream::binary);

		if (!ifs.is_open())
		{
			return false;
		}

		// stringstream serves as sink for the input filter
		std::stringstream str_buffer;

		// create input filter
		boost::iostreams::filtering_streambuf<boost::iostreams::input> in;

		// select filter according to file compression
		switch (ifs_compression)
		{
			case file_compression::gzip:
				in.push(boost::iostreams::gzip_decompressor());
				break;
			case file_compression::bzip2:
				in.push(boost::iostreams::bzip2_decompressor());
				break;
			case file_compression::none:
				break;
		}

		in.push(ifs);

		// copy data to stringbuffer
		boost::iostreams::copy(in, str_buffer);

		ifile = str_buffer.str();
		message_start = 0;
		message_end = 0;

		ifs.close();

		// return true if file opened succesfully
		return true;
	}

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

bool NFmiGrib::BuildIndex(const std::string &theFileName, const std::vector<std::string> &theKeys)
{
	std::string keyString;
	int err = 0;

	for (auto key : theKeys)
	{
		keyString.append(key);
		keyString.append(",");
	}

	keyString.pop_back();  // remove last comma

	index = grib_index_new(0, keyString.c_str(), &err);
	assert(index);
	GRIB_CHECK(grib_index_add_file(index, theFileName.c_str()), 0);

	return true;
}

bool NFmiGrib::BuildIndex(const std::string &theFileName, const std::string &theKeys)
{
	int err = 0;

	index = grib_index_new(0, theKeys.c_str(), &err);

	if (err)
	{
		return false;
	}

        assert(index);
	GRIB_CHECK(grib_index_add_file(index, theFileName.c_str()), 0);

	return true;
}

bool NFmiGrib::AddFileToIndex(const std::string &theFileName)
{
	assert(index);
	GRIB_CHECK(grib_index_add_file(index, theFileName.c_str()), 0);

	return true;
}

std::vector<long> NFmiGrib::GetIndexValues(const std::string &theKey)
{
	assert(index);

	size_t size = 0;
	std::vector<long> values;

	GRIB_CHECK(grib_index_get_size(index, theKey.c_str(), &size), 0);

	values.resize(size);

	GRIB_CHECK(grib_index_get_long(index, theKey.c_str(), values.data(), &size), 0);
	return values;
}

bool NFmiGrib::Message(const std::map<std::string, long> &theKeyValue)
{
	assert(index);
	int ret = 0;

	for (auto p : theKeyValue)
	{
		GRIB_CHECK(grib_index_select_long(index, (p.first).c_str(), p.second), 0);
	}

	h = grib_handle_new_from_index(index, &ret);

	if (ret == GRIB_END_OF_INDEX)
	{
		return false;
	}
	assert(h);
	return m.Read(h);
}

bool NFmiGrib::NextMessage()
{
	if (h)
	{
		// this invalidates itsHandle
		grib_handle_delete(h);
		h = 0;
	}

	assert(!h);

	/*
	 * Read data through filtering streambuf if gzip packed.
	 */

	if (ifs_compression == file_compression::gzip || ifs_compression == file_compression::bzip2)
	{
		message_start = ifile.find("GRIB", message_end);      // find next grib message after previous message ended
		message_end = ifile.find("GRIB", message_start + 1);  // find end of next grib message

		if (message_end == std::string::npos) message_end = ifile.size();

		size_t message_length = message_end - message_start;

		if (message_start == std::string::npos) return false;

		// create grib_message from stringbuffer
		if ((h = grib_handle_new_from_message_copy(0, (ifile.substr(message_start, message_length)).c_str(),
		                                           message_length * sizeof(char))) != NULL)
		{
			itsCurrentMessage++;
			assert(h);

			return m.Read(h);
		}

		return false;
	}

	int err;

	if ((h = grib_handle_new_from_file(0, f, &err)) != NULL)
	{
		itsCurrentMessage++;
		assert(h);
		return m.Read(h);
	}

	return false;
}

int NFmiGrib::MessageCount()
{
	if (ifs_compression != file_compression::none)
	{
		size_t index = -1;
		itsMessageCount = -1;

		do
		{
			index = ifile.find("GRIB", index + 1);
			++itsMessageCount;
		} while (index < std::string::npos);

		return itsMessageCount;
	}

	if (itsMessageCount == INVALID_INT_VALUE)
	{
		GRIB_CHECK(grib_count_in_file(0, f, &itsMessageCount), 0);
	}

	return itsMessageCount;
}

int NFmiGrib::CurrentMessageIndex() { return itsCurrentMessage; }
void NFmiGrib::MultiGribSupport(bool theMultiGribSupport)
{
	if (theMultiGribSupport)
		grib_multi_support_on(0);
	else
		grib_multi_support_off(0);
}

bool NFmiGrib::WriteMessage(const std::string &theFileName)
{
	// Assume we have required directory structure in place

	boost::filesystem::path p(theFileName);

	std::string ext = p.extension().string();

	// determine compression type for out file
	if (ext == ".gz")
	{
		ofs_compression = file_compression::gzip;
	}
	else if (ext == ".bz2")
	{
		ofs_compression = file_compression::bzip2;
	}
	else
	{
		ofs_compression = file_compression::none;
	}

	// write compressed output
	if (ofs_compression == file_compression::gzip || ofs_compression == file_compression::bzip2)
	{
		const void *buffer;
		size_t bfr_size;

		GRIB_CHECK(grib_get_message(h, &buffer, &bfr_size), 0);

		// copy data to stringstream as source for filtering ofstream
		std::string str_bfr(static_cast<const char *>(buffer), bfr_size);
		std::stringstream outdata(str_bfr);

		ofs.open(theFileName.c_str(), std::ofstream::out);
		boost::iostreams::filtering_streambuf<boost::iostreams::input> out;
		switch (ofs_compression)
		{
			case file_compression::gzip:
				out.push(boost::iostreams::gzip_compressor());
				break;
			case file_compression::bzip2:
				out.push(boost::iostreams::bzip2_compressor());
				break;
			case file_compression::none:
				break;
		}
		out.push(outdata);
		boost::iostreams::copy(out, ofs);

		return true;
	}

	GRIB_CHECK(grib_write_message(h, theFileName.c_str(), "w"), 0);

	return true;
}

bool NFmiGrib::WriteIndex(const std::string &theFileName)
{
	GRIB_CHECK(grib_index_write(index, theFileName.c_str()), 0);

	return true;
}
