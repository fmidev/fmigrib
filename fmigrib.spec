%define LIBNAME fmigrib
Summary: fmigrib library
Name: lib%{LIBNAME}
Version: 14.3.17
Release: 1.fmi
License: FMI
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Provides: %{LIBNAME}
BuildRequires: grib_api-devel >= 1.10.4
BuildRequires: boost-devel >= 1.54

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
%makeinstall

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
%{_libdir}/lib%{LIBNAME}.so.*

%files devel
%defattr(-,root,root,0644)
%{_libdir}/lib%{LIBNAME}.a
%{_includedir}/*.h

%changelog
* Mon Mar 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.17-1.fmi
- Changes in grib2 writing
* Fri Nov 29 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.29-1.fmi
- Do not count grib messages when opening file
- New features required by Harmonie 
* Tue Oct  3 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.3-1.fmi
- Initial build
