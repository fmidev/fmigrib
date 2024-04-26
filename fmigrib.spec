%if !0%{?version:1}
%define version 23.7.14
%endif

%if !0%{?release:1}
%define release 1
%endif

%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%if %{distnum} == 8
%define boost boost169
%else
%define boost boost
%endif

Summary: fmigrib library
Name: libfmigrib
Version: %{version}
Release: %{release}%{dist}.fmi
License: MIT
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Provides: fmigrib.so
BuildRequires: eccodes-devel
BuildRequires: python3-distro
BuildRequires: cuda-nvcc-12-4
BuildRequires: python3-scons
BuildRequires: cuda-cudart-devel-12-4
BuildRequires: %{boost}-devel
Buildrequires: zlib-devel
BuildRequires: bzip2-devel
BuildRequires: xz-devel
BuildRequires: make
BuildRequires: gcc-c++
Requires: %{boost}-iostreams
Requires: zlib
Requires: bzip2
Requires: xz-libs
Provides: libfmigrib.so

%description
FMI grib library

%package devel
Summary: development package
Group: Development/Tools

%description devel
Headers and static libraries for fmigrib

%prep
rm -rf $RPM_BUILD_ROOT

%setup -q -n "libfmigrib" 

%build
make %{_smp_mflags} 

%install
mkdir -p $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT

%post
umask 007
/sbin/ldconfig > /dev/null 2>&1

%postun
umask 007
/sbin/ldconfig > /dev/null 2>&1

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,0644)
%{_libdir}/libfmigrib.so*

%files devel
%defattr(-,root,root,0644)
%{_libdir}/libfmigrib.a
%{_includedir}/*.h

%changelog
* Fri Jul 14 2023 Mikko Partio <mikko.partio@fmi.fi> - 23.7.14-1.fmi
- Improved error handling
* Thu Mar 16 2023 Mikko Partio <mikko.partio@fmi.fi> - 23.3.16-1.fmi
- Support typeOfGeneratingProcess=11
* Mon Feb  6 2023 Mikko Partio <mikko.partio@fmi.fi> - 23.2.6-1.fmi
- Link against boost thread
* Fri Jan 27 2023 Mikko Partio <mikko.partio@fmi.fi> - 23.1.27-1.fmi
- Add function to delete a handle
* Thu Jan 26 2023 Mikko Partio <mikko.partio@fmi.fi> - 23.1.26-1.fmi
- Link explicitly with boost filesystem
* Wed Dec  2 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.12.2-1.fmi
- Correct counting of grib messages on padded files
* Tue Nov  3 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.11.3-1.fmi
- Better to way to deal with errors when counting messages
* Fri Oct 16 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.10.16-1.fmi
- Fix race condition bug
* Wed Oct 14 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.10.14-1.fmi
- Prevent integer overflow with large grib messages
* Mon Oct  5 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.10.5-1.fmi
- Alternative way to calculate message position
* Tue Jun 16 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.6.16-1.fmi
- Minor bugfix for empty grib handling
* Mon May 25 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.5.25-1.fmi
- Function additions
* Mon Apr 20 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.4.20-1.fmi
- boost 1.69
* Wed Mar 18 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.3.18-1.fmi
- Add function to read gribs from file pointer
* Mon Mar 16 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.3.16-1.fmi
- Tuning grib unpacking
* Wed Feb  5 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.2.5-1.fmi
- Read/write forecast_type_id 5
* Fri Jan 31 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.1.31-1.fmi
- Fix memory leak
* Mon Jan 20 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.1.20-1.fmi
- Add function to get grib message from memory
* Wed Nov 20 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.11.20-1.fmi
- Compile for compute capability 7.0
* Wed Nov  6 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.11.6-1.fmi
- Bugfix
* Mon Nov  4 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.11.4-1.fmi
- Read message directly from memory
* Mon Oct 28 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.10.28-1.fmi
- Packing/unpacking as template functions
* Fri Oct 25 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.10.25-1.fmi
- Merge unpacking code from Himan
* Wed Oct 23 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.10.23-1.fmi
- More fixes to Cuda packing
* Mon Oct 21 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.10.21-1.fmi
- Fixes to Cuda packing
* Fri Sep 20 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.9.20-1.fmi
- HIMAN-283: store message offsets, support reading from a position
* Mon Jun 17 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.6.17-1.fmi
- Fix for MNWC accumulated parameter time unit
* Wed Jun 12 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.6.12-1.fmi
- MNWC has indicatorOfUnitOfTimerange=14
* Wed Apr 24 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.4.24-1.fmi
- Add maxwind level to level map
* Thu Apr  4 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.4.4-1.fmi
- Fix for grib2 minute timestep handling
* Tue Feb 12 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.2.12-1.fmi
- Overload for fetching grib index
* Thu Oct  4 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.10.4-1.fmi
- Fix constant grid filling with bitmap on
* Wed Jun 13 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.6.13-1.fmi
- Supporting minute resolution with grib2
* Tue May 15 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.5.15-1.fmi
- Bugfix
* Thu May  3 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.5.3-1.fmi
- Support reading from stdin
* Tue Apr 10 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.4.10-1.fmi
- New boost
* Mon Feb 12 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.2.12-1.fmi
- const change on PV() function
- new constructor for NFmiGribMessage
* Wed Jan 24 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.1.24-1.fmi
- ENTATM to level map
* Mon Dec 11 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.12.11-1.fmi
- New leveltype pressure delta
* Wed Oct 25 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.10.25-1.fmi
- Remove grib_handle handling from NFmiGrib
* Wed Sep 27 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.9.27-1.fmi
- Bugfixes in level handling
* Wed Aug  2 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.2-1.fmi
- eccodes 2.4.0
* Mon Jul 31 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.7.31-1.fmi
- Add function to get grib values without extra allocations
* Thu Apr  6 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.6-1.fmi
- Remove obsolete code
* Wed Dec  7 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.7-1.fmi
- Cuda 8.0
* Wed Nov  2 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.2-1.fmi
- Minor fixes
* Tue Oct 25 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.25-1.fmi
- Support ECMWF seasonal forecast
* Thu Oct 20 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.20-1.fmi
- Replacing grib_api with eccodes
* Thu Sep  8 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.9.8-1.fmi
- More support reading grib index files
* Mon Aug 15 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.8.15-1.fmi
- Support reading grib index files
* Mon Jun 13 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.13-2.fmi
- Fix for ECMWF 06/18 which are kind of ensemble but really not 
* Mon Jun 13 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.13-1.fmi
- New release
* Thu Jun  2 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.2-1.fmi
- New release
* Wed Jun  1 2016 Mikko Aalto <mikko.aalto@fmi.fi> - 16.6.1-1.fmi
- grib_api 1.15
* Tue May 24 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.5.24-1.fmi
- EPS fixes
* Tue Feb 23 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.23-1.fmi
- New release
* Fri Feb 12 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.12-1.fmi
- Fix end step for grib2 accumulation parameters
* Mon Sep 14 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.14-1.fmi
- Two cuda device memory leaks fixed
* Fri Sep 11 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.11-1.fmi
- Fix to cuda unpack with bitmap
* Thu Sep 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.10-1.fmi
- Bugfix for static grids
* Wed Sep  9 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.9-1.fmi
- Fix to cuda unpack when memory was free'd before read
* Thu Sep  3 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.3-1.fmi
- Fix to cuda unpack with bitmap
* Wed Sep  2 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.2-1.fmi
- grib_api 1.14
* Fri Aug 21 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.8.21-1.fmi
- Add functions to get and set grib missing value
* Mon May 18 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.5.18-1.fmi
- add compression feature to NFmiGribMessage
* Tue May 12 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.5.12-1.fmi
- bugfix for compression features
* Thu May  7 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.5.7-1.fmi
- gz and bzip2 compression features added
* Thu Apr 16 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.16-1.fmi
- Simplifies internal structure
* Wed Apr  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.8-1.fmi
- Reworked ForecastType logic
* Tue Mar 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.3.10-1.fmi
- Add support for cloning NFmiGribMessage
* Mon Mar  9 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.3.9-1.fmi
- Packing and unpacking of simple packing and jpeg packing moved from himan to fmigrib
* Tue Nov 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.18-1.fmi
- Add support for writing packed data
* Thu Oct  9 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.9-1.fmi
- Fix in NormalizedStep() and grib2
* Wed Oct  8 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.8-1.fmi
- Add Type() function
* Mon Sep 29 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.29-1.fmi
- Do not create grib_handle until it is necessary to avoid random crashes (ECMWF: SUP-1023)
* Mon Mar 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.17-1.fmi
- Changes in grib2 writing
* Fri Nov 29 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.29-1.fmi
- Do not count grib messages when opening file
- New features required by Harmonie 
* Tue Oct  3 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.3-1.fmi
- Initial build
