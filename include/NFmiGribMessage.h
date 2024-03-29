/*
 * NFmiGribMessage.h
 *
 * One NFmiGribMessage equals to one grib message.
 * All the setter functions modify in-memory structures,
 * the changes are are materialized only when the message
 * is written to disk with WriteMessage().
 */

#ifndef NFMIGRIBMESSAGE_H_
#define NFMIGRIBMESSAGE_H_

#include <grib_api.h>
#include <map>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#if defined __GNUC__
#if __GNUC_MINOR__ > 5
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

#include <cuda_runtime.h>

#if defined __GNUC__ && __GNUC_MINOR__ > 5
#pragma GCC diagnostic pop
#endif
#endif

class NFmiGribMessage
{
   public:
	NFmiGribMessage();
	NFmiGribMessage(void* buf, long size);

	~NFmiGribMessage();

	NFmiGribMessage(const NFmiGribMessage& other);
	NFmiGribMessage& operator=(const NFmiGribMessage& other);

	void DeleteHandle();
	bool Read(grib_handle** h);

	long SizeX() const;
	long SizeY() const;

	void SizeX(long theXSize);
	void SizeY(long theYSize);

	double X0() const;
	double Y0() const;

	void X0(double theX0);
	void Y0(double theY0);

	double X1() const;
	double Y1() const;

	void X1(double theX1);
	void Y1(double theY1);

	double SouthPoleX() const;
	void SouthPoleX(double theLongitude);

	double SouthPoleY() const;
	void SouthPoleY(double theLatitude);

	[[deprecated("Use GetValues()")]] double* Values() const;
	void GetValues(double* values, size_t* cntValues) const;
	void Values(const double* theValues, long theValuesLength);

	[[deprecated("Use GetValues()")]] double* Values(double missingValue) const;
	void GetValues(double* values, size_t* cntValues, double missingValue) const;
	void Values(const double* theValues, long theValuesLength, double missingValue);

	size_t ValuesLength() const;

	long DataDate() const;
	void DataDate(long theDate);

	long DataTime() const;
	void DataTime(long theTime);

	long ForecastTime() const;
	void ForecastTime(long theTime);

	std::string ParameterUnit() const;
	long ParameterNumber() const;
	long ParameterDiscipline() const;
	long ParameterCategory() const;

	void ParameterNumber(long theNumber);
	void ParameterDiscipline(long theDiscipline);
	void ParameterCategory(long theCategory);

	std::string ParameterName() const;

	long GridType() const;
	void GridType(long theGridType);

	long Edition() const;
	void Edition(long theEdition);

	double GridOrientation() const;
	void GridOrientation(double theGridOrientation);

	long Centre() const;
	void Centre(long theCentre);

	long Year() const;
	long Month() const;
	long Day() const;
	long Hour() const;
	long Minute() const;
	long Second() const;

	long Process() const;
	void Process(long theProcess);

	long Table2Version() const;
	void Table2Version(long theVersion);

	long PerturbationNumber() const;
	void PerturbationNumber(long thePerturbationNumber);

	long NormalizedGridType(unsigned int targetEdition = 1) const;
	long NormalizedLevelType(unsigned int targetEdition = 1) const;

	long LocalDefinitionNumber() const;

	long StartStep() const;
	void StartStep(long theStartStep);

	long EndStep() const;
	void EndStep(long theEndStep);

	long StepUnits() const;
	void StepUnits(long theUnits);

	long StepRange() const;
	void StepRange(long theRange);

	long TimeRangeIndicator() const;
	void TimeRangeIndicator(long theTimeRangeIndicator);

	bool Write(const std::string& theOutputFile, bool appendToFile = false);

	void Year(const std::string& theYear);
	void Month(const std::string& theMonth);
	void Day(const std::string& theDay);
	void Hour(const std::string& theHour);
	void Minute(const std::string& theMinute);
	void Second(const std::string& theSecond);

	bool Bitmap() const;
	void Bitmap(bool theBitmap);

	long BitsPerValue() const;
	void BitsPerValue(long theBitsPerValue);

	void PackingType(const std::string& thePackingType);
	std::string PackingType() const;

	long LevelValue() const;
	void LevelValue(long theLevelValue, long theScaleFactor = 0);

	long LevelValue2() const;
	void LevelValue2(long theLevelValue2, long theScaleFactor = 0);

	long LevelType() const;
	void LevelType(long theLevelType);

	bool IScansNegatively() const;
	bool JScansPositively() const;

	void IScansNegatively(bool theNegativeIScan);
	void JScansPositively(bool thePositiveJScan);

	double iDirectionIncrement() const;
	void iDirectionIncrement(double theIncrement);

	double jDirectionIncrement() const;
	void jDirectionIncrement(double theIncrement);

	long TypeOfGeneratingProcess() const;
	void TypeOfGeneratingProcess(long theProcess);

	void XLengthInMeters(double theLength);
	void YLengthInMeters(double theLength);

	double XLengthInMeters() const;
	double YLengthInMeters() const;

	long GridTypeToAnotherEdition(long gridType, long edition) const;
	long LevelTypeToAnotherEdition(long levelType, long edition) const;

	long NumberOfMissing() const;

	double MissingValue() const;

	// Note: this function is const although it set the missing value.
	// We need it because while fetching data, we might alter the missing value.
	void MissingValue(double missingValue) const;

	bool UVRelativeToGrid() const;
	void UVRelativeToGrid(bool theRelativity);

	long UnitOfTimeRange() const;
	void UnitOfTimeRange(long theUnit);

	long LengthOfTimeRange() const;
	void LengthOfTimeRange(long theLength);

	long P1() const;
	void P1(long theP1);

	long P2() const;
	void P2(long theP2);

	long ResolutionAndComponentFlags() const;
	void ResolutionAndComponentFlags(long theResolutionAndComponentFlags);

	long NV() const;
	void NV(long theNV);

	std::vector<double> PV() const;
	std::vector<double> PV(size_t theNumberOfCoordinates, size_t level);
	void PV(const std::vector<double>& theAB, size_t abLen);

	size_t PackedValuesLength() const;

	bool PackedValues(unsigned char* data) const;
	bool PackedValues(unsigned char* data, size_t unpacked_len, int* bitmap, size_t bitmap_len);

	long BinaryScaleFactor() const;
	void BinaryScaleFactor(long theFactor);

	long DecimalScaleFactor() const;
	void DecimalScaleFactor(long theFactor);

	void ChangeDecimalPrecision(long decimals);

	double ReferenceValue() const;
	void ReferenceValue(double theValue);

	long Section4Length() const;

	size_t BytesLength(const std::string& key) const;
	bool Bytes(const std::string& key, unsigned char* data) const;

	long ProductDefinitionTemplateNumber() const;
	void ProductDefinitionTemplateNumber(long theNumber);

	long TypeOfStatisticalProcessing() const;
	void TypeOfStatisticalProcessing(long theType);

	// grib_api version > 1.10.4
	bool KeyExists(const std::string& theKey) const;

	std::vector<int> PL() const;
	void PL(const std::vector<int> thePL);

	/**
	 * @brief Normalizing unitOfTimeRange to GRIB 1 values
	 */

	long NormalizedUnitOfTimeRange() const;

	/**
	 * @brief Return "flattened" value of start step or end step
	 *
	 * For example unit for time range could be 15 minutes, and time step in
	 * grib could have value 3. This function would then return value 45.
	 *
	 * Function is normalized meaning it will return values in GRIB1 style
	 * even though data is in GRIB2.
	 *
	 * @param endStep If true, return value of end step. Otherwise return value of start step.
	 * @param flatten If true, time value is flattened as described above
	 * @return Flattened time step value
	 */

	long NormalizedStep(bool endStep, bool flatten) const;

	long Type() const;
	void Type(long theType);

	template <typename T>
	bool CudaUnpack(T* arr, size_t len);
	template <typename T>
	bool CudaPack(T* arr, size_t len);

#ifdef HAVE_CUDA
	template <typename T>
	bool CudaUnpack(T* arr, size_t len, cudaStream_t& stream);
	template <typename T>
	bool CudaPack(T* arr, size_t len, cudaStream_t& stream);
#endif

	double CalculateReferenceValue(double minimumValue);

	long ForecastType() const;
	void ForecastType(long theForecastType);

	long ForecastTypeValue() const;
	void ForecastTypeValue(long theForecastTypeValue);

	long GetLongKey(const std::string& keyName) const;
	void SetLongKey(const std::string& keyName, long value);

	double GetDoubleKey(const std::string& keyName) const;
	void SetDoubleKey(const std::string& keyName, double value);

	std::string GetStringKey(const std::string& keyName) const;

	void GetMessage(unsigned char* content, size_t length);

	void Repack();

   private:
	void Clear();

	size_t GetSizeTKey(const std::string& keyName) const;
	grib_handle* itsHandle;

	mutable long itsEdition;  //<! Cache this key since it's used quite a lot

	mutable size_t itsPackedValuesLength;
};

#endif /* NFMIGRIBMESSAGE_H_ */
