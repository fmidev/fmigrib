#
# SConscript for himan-lib

import os
import platform
import sys

OS_NAME = platform.linux_distribution()[0]
OS_VERSION = float(platform.linux_distribution()[1])

IS_RHEL = False

if OS_NAME == "Red Hat Enterprise Linux Server" or OS_NAME == "CentOS":
	IS_RHEL=True

AddOption(
    '--debug-build',
    dest='debug-build',
    action='store_true',
    help='debug build',
    default=False)

AddOption(
    '--test-build',
    dest='test-build',
    action='store_true',
    help='test build',
    default=False)

env = Environment()

# Check build

DEBUG = GetOption('debug-build')
TEST = GetOption('test-build')
RELEASE = (not DEBUG)

if TEST:
	DEBUG = True

# Workspace

workspace = os.getcwd() + "/../"

# cuda toolkit path

cuda_toolkit_path = '/usr/local/cuda-6.5'

if os.environ.get('CUDA_TOOLKIT_PATH') is None:
        print "Environment variable CUDA_TOOLKIT_PATH not set, assuming " + cuda_toolkit_path
else:
        cuda_toolkit_path = os.environ['CUDA_TOOLKIT_PATH']

have_cuda = False

if os.path.isfile(cuda_toolkit_path + '/lib64/libcudart.so'):
        have_cuda = True

have_comprimato = False

if os.path.isfile('/usr/lib64/libcmpto_j2k_dec.so'):
        have_comprimato = True

if os.environ.get('CC') != None:
        env['CC'] = os.environ.get('CC')
else:
	env['CC'] = 'gcc'

if os.environ.get('CXX') != None:
        env['CXX'] = os.environ.get('CXX')
else:
	env['CXX'] = 'g++'

# Includes

includes = []

includes.append('./include')

if have_cuda:
        includes.append(cuda_toolkit_path + '/include')

env.Append(CPPPATH = includes)

# Library paths

librarypaths = []

librarypaths.append('/usr/lib64')

env.Append(LIBPATH = librarypaths)

# Libraries

libraries = []

env.Append(LIBS = libraries)

if IS_RHEL and OS_VERSION < 7.0:
	boost_libraries = [ 'boost_program_options', 'boost_filesystem', 'boost_system', 'boost_regex', 'boost_iostreams', 'boost_thread' ]

	for lib in boost_libraries:
		libfile = '/usr/lib64/lib' + lib + '.a'

		if not os.path.isfile(libfile):
			libfile = '/usr/lib64/lib' + lib + '-mt.a'

		env.Append(LIBS=env.File(libfile))

if have_cuda:
	env.Append(LIBS=env.File(cuda_toolkit_path + '/lib64/libcudart_static.a'))

# CFLAGS

# "Normal" flags

cflags_normal = []
cflags_normal.append('-Wall')
cflags_normal.append('-W')
cflags_normal.append('-Wno-unused-parameter')
cflags_normal.append('-Werror')

# Extra flags

cflags_extra = []
cflags_extra.append('-Wpointer-arith')
cflags_extra.append('-Wcast-qual')
cflags_extra.append('-Wcast-align')
cflags_extra.append('-Wwrite-strings')
cflags_extra.append('-Wconversion')
cflags_extra.append('-Winline')
cflags_extra.append('-Wnon-virtual-dtor')
cflags_extra.append('-Wno-pmf-conversions')
cflags_extra.append('-Wsign-promo')
cflags_extra.append('-Wchar-subscripts')
cflags_extra.append('-Wold-style-cast')

# Difficult flags

cflags_difficult = []
cflags_difficult.append('-pedantic')
cflags_difficult.append('-Weffc++')
cflags_difficult.append('-Wredundant-decls')
cflags_difficult.append('-Wshadow')
cflags_difficult.append('-Woverloaded-virtual')
cflags_difficult.append('-Wunreachable-code')
cflags_difficult.append('-Wctor-dtor-privacy')

# Default flags (common for release/debug)

cflags = []

if not IS_RHEL or (IS_RHEL and OS_VERSION >= 7.0):
	cflags.append('-std=c++0x')
else:
	cflags.append('-std=c++0x')

cflags.append('-fPIC')

env.Append(CCFLAGS = cflags)
env.Append(CCFLAGS = cflags_normal)

# Linker flags

env.Append(LINKFLAGS = ['-rdynamic','-Wl,--as-needed'])

# Defines

env.Append(CPPDEFINES=['UNIX'])

if have_cuda:
        env.Append(CPPDEFINES=['HAVE_CUDA'])

if have_comprimato:
        env.Append(CPPDEFINES=['HAVE_COMPRIMATO'])

env.Append(NVCCDEFINES=['HAVE_CUDA'])

env.Append(NVCCFLAGS = ['-m64'])
env.Append(NVCCFLAGS = ['-Xcompiler','-fPIC'])
env.Append(NVCCFLAGS = ['-Xcompiler','-Wall'])
env.Append(NVCCFLAGS = ['-arch=compute_35','-code=sm_35'])

if IS_RHEL and OS_VERSION >= 7.0:
	env.Append(NVCCFLAGS = ['-std=c++11'])

env.Append(NVCCPATH = ['./include'])

# Other

build_dir = ""

if RELEASE:
	env.Append(CCFLAGS = ['-O2'])
	env.Append(CPPDEFINES = ['NDEBUG'])
	build_dir = 'build/release'

if DEBUG:
	env.Append(CCFLAGS = ['-g','-O0'])
	env.Append(CCFLAGS = cflags_extra)
	#env.Append(CCFLAGS = cflags_difficult)
	env.Append(CPPDEFINES = ['DEBUG'])

	# Cuda
	env.Append(NVCCFLAGS = ['-O0','-g','-G', '-lineinfo'])
	env.Append(NVCCDEFINES = ['DEBUG'])

	build_dir = 'build/debug'

if TEST:
	build_dir = 'build/debug'

	env['IMPLICIT_COMMAND_DEPENDENCIES'] = 0
	env.Append(LIBS = ['fmigrib', 'boost_unit_test_framework'])
	SConscript('SConscript_test', exports = ['env'], variant_dir=build_dir, duplicate=1)
#	sys.exit(0)

SConscript('SConscript', exports = ['env'], variant_dir=build_dir, duplicate=0)
Clean('.', build_dir)