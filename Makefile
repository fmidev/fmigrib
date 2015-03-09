LIB = fmigrib

SCONS_FLAGS=-j 4

# How to install

INSTALL_DATA = install -m 664

#rpmsourcedir = /home/partio/rpmbuild/SOURCES
rpmsourcedir=/tmp/$(shell whoami)/rpmbuild

INSTALL_TARGET = /usr/lib64

# The rules

all release: 
	scons $(SCONS_FLAGS)
debug: 
	scons $(SCONS_FLAGS) --debug-build
test:
	scons $(SCONS_FLAGS) --debug-build --test-build
clean:
	scons -c ; scons --debug-build -c ; rm -f *~ source/*~ include/*~

rpm:    clean
	mkdir -p $(rpmsourcedir) ; \
	if [ -a $(LIB).spec ]; \
        then \
          tar -C ../ --exclude .svn \
                   -cf $(rpmsourcedir)/lib$(LIB).tar $(LIB) ; \
          gzip -f $(rpmsourcedir)/lib$(LIB).tar ; \
          rpmbuild -ta $(rpmsourcedir)/lib$(LIB).tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p $(DESTDIR)/$(INSTALL_TARGET) $(DESTDIR)/usr/include
	if [ -f "build/release/lib$(LIB).so" ]; then \
		$(INSTALL_DATA) build/release/lib$(LIB).so $(DESTDIR)/$(INSTALL_TARGET); \
		$(INSTALL_DATA) build/release/lib$(LIB).a $(DESTDIR)/$(INSTALL_TARGET); \
		$(INSTALL_DATA) include/NFmiGribPacking.h $(DESTDIR)/usr/include; \
		$(INSTALL_DATA) include/NFmiGrib.h $(DESTDIR)/usr/include; \
		$(INSTALL_DATA) include/NFmiGribMessage.h $(DESTDIR)/usr/include; \
	fi;

