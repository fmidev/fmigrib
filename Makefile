LIB = fmigrib

SCONS_FLAGS=-j 4

# How to install

INSTALL_DATA = install -m 644
INSTALL_LIB = install -m 755

#rpmsourcedir = /home/partio/rpmbuild/SOURCES
rpmsourcedir=/tmp/$(shell whoami)/rpmbuild

INSTALL_TARGET = /usr/lib64

# The rules

all release: 
	scons-3 $(SCONS_FLAGS)
debug: 
	scons-3 $(SCONS_FLAGS) --debug-build
test:
	cd tests && make
clean:
	scons-3 -c ; scons-3 --debug-build -c ; rm -f *~ source/*~ include/*~

rpm:    clean $(LIB).spec
	mkdir -p $(rpmsourcedir)
	tar --transform "s,^./,libfmigrib/,"  --exclude-vcs -czf $(rpmsourcedir)/lib$(LIB).tar.gz .
	rpmbuild -ta $(rpmsourcedir)/lib$(LIB).tar.gz

install:
	mkdir -p $(DESTDIR)/$(INSTALL_TARGET) $(DESTDIR)/usr/include
	if [ -f "build/release/lib$(LIB).so" ]; then \
		$(INSTALL_LIB) build/release/lib$(LIB).so $(DESTDIR)/$(INSTALL_TARGET); \
		$(INSTALL_LIB) build/release/lib$(LIB).a $(DESTDIR)/$(INSTALL_TARGET); \
		$(INSTALL_DATA) include/NFmiGribPacking.h $(DESTDIR)/usr/include; \
		$(INSTALL_DATA) include/NFmiGrib.h $(DESTDIR)/usr/include; \
		$(INSTALL_DATA) include/NFmiGribMessage.h $(DESTDIR)/usr/include; \
	fi;

