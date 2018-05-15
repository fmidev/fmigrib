%define LIBNAME fmigrib
Summary: fmigrib library
Name: lib%{LIBNAME}
Version: 18.5.3
Release: 1.el7.fmi
License: MIT
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Provides: %{LIBNAME}.so
BuildRequires: eccodes-devel
BuildRequires: boost-devel >= 1.66
Provides: lib%{LIBNAME}.so

%description
FMI grib library

%package devel
Summary: development package
Group: Development/Tools

%description devel
Headers and static libraries for fmigrib

%prep
rm -rf $RPM_BUILD_ROOT

%setup -q -n "%{LIBNAME}" 

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
%{_libdir}/lib%{LIBNAME}.so*

%files devel
%defattr(-,root,root,0644)
%{_libdir}/lib%{LIBNAME}.a
%{_includedir}/*.h

%changelog
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
