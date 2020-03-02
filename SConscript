#
# SConstruct for himan-lib

Import('env')
import os
import sys

# cuda toolkit path

cuda_toolkit_path = '/usr/local/cuda'

if os.environ.get('CUDA_TOOLKIT_PATH') is not None:
        cuda_toolkit_path = os.environ['CUDA_TOOLKIT_PATH']

have_cuda = False

if os.path.isfile(cuda_toolkit_path + '/lib64/libcudart.so'):
        have_cuda = True

if have_cuda:
        env.Tool('cuda')

objects = []

for file in Glob('source/*.cpp'):
    s=os.path.basename(str(file))
    obj='obj/'+ s.replace(".cpp","")
    objects += env.SharedObject(obj, file)

if have_cuda:
    for file in Glob('source/*.cu'):
        s=os.path.basename(str(file))
        obj='obj/cu_'+ s.replace(".cu","")
        objects += env.SharedObject(obj, file)

lib = env.StaticLibrary(directory = 'lib', target = 'fmigrib', source = objects)

env.Install(dir = '/usr/lib64', source = lib)

lib = env.SharedLibrary(directory = 'lib', target = 'fmigrib', source = objects)

env.Install(dir = '/usr/lib64', source = lib)

