from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools
import platform
import glob

__version__ = '0.0.4.10'

# Adding chi_mapping_tool_stuff
lsdtt_list_so_far = ["src/LSDMostLikelyPartitionsFinder.cpp","src/LSDIndexRaster.cpp",
"src/LSDRaster.cpp","src/LSDRasterInfo.cpp","src/LSDFlowInfo.cpp",
"src/LSDJunctionNetwork.cpp","src/LSDIndexChannel.cpp","src/LSDChannel.cpp",
"src/LSDIndexChannelTree.cpp","src/LSDStatsTools.cpp","src/LSDShapeTools.cpp",
"src/LSDChiNetwork.cpp","src/LSDBasin.cpp","src/LSDParticle.cpp","src/LSDChiTools.cpp",
"src/LSDParameterParser.cpp","src/LSDSpatialCSVReader.cpp","src/LSDCRNParameters.cpp",
"src/LSDRasterMaker.cpp"]

#Adding the channel extraction stuff
lsdtt_list_so_far.append("src/LSDCosmoData.cpp")
# lsdtt_list_so_far.append("LSDCloudBase.cpp")
# lsdtt_list_so_far.append("LSDCloudRaster.cpp")

# Adding the python stuff
lsdtt_list_so_far.append('src/main.cpp')
lsdtt_list_so_far.append("src/LSD_xtensor_utils.cpp")
lsdtt_list_so_far.append("src/LSDEntry_points.cpp")
lsdtt_list_so_far.append("src/LSD_xtensor_convtools.cpp")
# lsdtt_list_so_far.append("src/LSDTribBasModel.cpp")
lsdtt_list_so_far.append("src/LSDDEM_xtensor.cpp")
# Alright these might crash at some point on windows but it worth a try innit?
libdir = []
lflags = []


if(platform.system() == 'Windows'):
    # Gathering the pcl stuff here
    # templib = glob.glob("C:/Users/s1675537/AppData/Local/Continuum/anaconda3/envs/lsdtopytools/Library/lib/pcl*")
    liraries_lsdttxtp = []
    # for i in templib:
    #     liraries_lsdttxtp.append(i.split("\\")[-1][:-4])


    # liraries_lsdttxtp.append("fftw3")
    flags = ["/Ox"]
    # lflags = ['/LIBPATH:C:/Users/s1675537/AppData/Local/Continuum/anaconda3/envs/lsdtopytools/Library/lib']
    # libdir = ['/LIBPATH:C:/Users/s1675537/AppData/Local/Continuum/anaconda3/envs/lsdtopytools/Library/lib']

else:
    # You have to specify if you want to install muddpyle
    if("muddpyle" in sys.argv):
        lsdtt_list_so_far.append("src/LSDRasterSpectral.cpp")
        lsdtt_list_so_far.append("src/LSDRasterModel.cpp")
        lsdtt_list_so_far.append("src/LSDParticleColumn.cpp")

    flags = ["-O3"]
    # lflags = ["-O3", "-L/lib64","-L/lib", "-lfftw3"]
    lflags = ["-O3"]
    liraries_lsdttxtp = [ "fftw3"]
    libdir = ["lib","/lib64"]







class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


class get_numpy_include(object):
    """Helper class to determine the numpy include path

    The purpose of this class is to postpone importing numpy
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self):
        pass

    def __str__(self):
        import numpy as np
        return np.get_include()


ext_modules = [
    Extension(
        'lsdtt_xtensor_python',
        lsdtt_list_so_far,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            get_numpy_include(),
            os.path.join(sys.prefix, 'include'),
            os.path.join(sys.prefix, 'Library', 'include'),
            "include",
            'src',
            'src/TNT'
        ],
        libraries = liraries_lsdttxtp,
        library_dirs = libdir,
        language='c++',
        extra_compile_args = flags,
        extra_link_args = lflags

    ),
]


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++14 compiler flag  and errors when the flag is
    no available.
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    else:
        raise RuntimeError('C++14 support is required by xtensor!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', "/Ox"],
        'unix': ["-O3"],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
            # if has_flag(self.compiler, '-fopenmp'):
            #     opts.append('-fopenmp')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='lsdtt_xtensor_python',
    version=__version__,
    author='LSDTopoTools team -  Boris Gailleton',
    author_email='boris.gailleton@gfz-potsdam.de',
    url='https://lsdtopotools.github.io',
    description= 'This is an attempt to port LSDTT to Python',
    long_description='''
lsdtt_xtensor_python is a python binder for LSDTopoTools. It directly binds in-memory the c++ code in order to use it alongside with python. 
Not all the routines are binded yet as I am still working on it. See lsdviztools for a full use of all the tools with command line/driver files/indirect python bindings.
Note that this is an intermediate package: lsdtopotools is the c++ package, lsdtt_xtensor_python offers low-level bindings to lsdtopotools and finally lsdtopytools offers the automated user-friendly scripting use.
''',
    keywords='lsdtt_xtensor_python',
    python_requires='>=3.7',
    license = "GNU General Public License v3",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.0.1', 'numpy'],
    cmdclass={'build_ext': BuildExt},
    data_files = [("", ["LICENSE"])],
    zip_safe=False,
)
