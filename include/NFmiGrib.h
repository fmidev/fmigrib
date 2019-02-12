/*
 * class NFmiGrib
 *
 * C++ overcoat for reading grib data and metadata.
 */

#ifndef __NFMIGRIB_H__
#define __NFMIGRIB_H__

#include "NFmiGribMessage.h"
#include <fstream>
#include <grib_api.h>
#include <memory>
#include <string>
#include <vector>

const int INVALID_INT_VALUE = -999;

class NFmiGrib
{
   public:
	NFmiGrib();
	~NFmiGrib();

	bool Open(const std::string& theFileName);
	bool BuildIndex(const std::string& theFileName, const std::vector<std::string>& theKeys);
	bool BuildIndex(const std::string& theFileName, const std::string& theKeys);

	bool AddFileToIndex(const std::string& theFileName);

	std::vector<long> GetIndexValues(const std::string& theKey);

	// Return message from index that match given keys. All keys must match.
	bool Message(const std::map<std::string, long>& theKeyValue);

	// Return message from index that matches keys in indexKeys and gribKeys.
	// indexKeys must be in the grib index
	// all messages (might be more than one) found from index are then searched
	// with gribKeys, and first message matching them is returned
	// this is a workaround for the fact that it's practically impossible to create
	// a grib index that support both grib editions
	bool Message(const std::map<std::string, long>& indexKeys, const std::map<std::string, long>& gribKeys);

	bool NextMessage();
	int MessageCount();
	int CurrentMessageIndex();

	void MultiGribSupport(bool theMultiGribSupport);
	bool WriteMessage(const std::string& theFileName);
	bool WriteIndex(const std::string& theFileName);

	NFmiGribMessage& Message()
	{
		return itsMessage;
	}

   private:
	enum class file_compression
	{
		none,
		gzip,
		bzip2
	};
	file_compression ifs_compression;
	file_compression ofs_compression;
	std::ifstream ifs;
	std::ofstream ofs;

	// string serves as sink for the input filter
	std::string ifile;
	size_t message_start;
	size_t message_end;

	grib_index* index;
	FILE* f;

	int itsMessageCount;
	int itsCurrentMessage;

	NFmiGribMessage itsMessage;
};

#endif
