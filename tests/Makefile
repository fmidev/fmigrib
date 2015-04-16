SCONS_FLAGS=-j 4

# How to install

INSTALL_DATA = install -m 664

#rpmsourcedir = /home/partio/rpmbuild/SOURCES
rpmsourcedir=/tmp/$(shell whoami)/rpmbuild

INSTALL_TARGET = /usr/lib64

# The rules

all release: 
	scons $(SCONS_FLAGS)
	sh run.sh build/release
debug: 
	scons $(SCONS_FLAGS) --debug-build
	sh run.sh build/debug
clean:
	scons -c ; scons --debug-build -c ; rm -f *~ source/*~ include/*~
