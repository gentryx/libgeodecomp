#============= 0. GENERIC CONFIGURATION ==============================
cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

set(PACKAGE_NAME libgeodecomp)
set(PACKAGE_VERSION "0.4.0")
set(PACKAGE_VENDOR "Ste||ar Group")
set(PACKAGE_HOMEPAGE "http:\\\\www.libgeodecomp.org")
set(PACKAGE_EMAIL "users@libgeodecomp.org")
project(${PACKAGE_NAME})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(util)
include(CheckCXXCompilerFlag)

if (CMAKE_VERSION VERSION_GREATER 2.7)
  cmake_policy(SET CMP0012 NEW)
endif()

# this is a workaround for CMake complaining about the interface
# include path of geodecomp.lib including the current build
# directory.
if(WIN32)
  cmake_policy(SET CMP0041 OLD)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

get_directory_property(lgd_has_parent PARENT_DIRECTORY)

#============= 1. FIND PACKAGES WE DEPEND ON =========================

find_package(HPX QUIET)

if(NOT HPX_FOUND)
  if(NOT DEFINED BOOST_ROOT)
    # deduce Boost location from environment (e.g. on woody.rrze.uni-erlangen.de)
    set(BOOST_ENV_LIBDIR "$ENV{BOOST_LIBDIR}")
    if(BOOST_ENV_LIBDIR)
      set(Boost_NO_SYSTEM_PATHS true)
      set(BOOST_ROOT "$ENV{BOOST_LIBDIR}/../")
      set(Boost_LIBRARY_DIRS "$ENV{BOOST_LIBDIR}")
      set(Boost_INCLUDE_DIRS "$ENV{BOOST_INCDIR}")
    endif()
  endif()

  # The reason for this ugly multi-stage discovery is the following:
  # since CMAKE 2.8.11 each call to find_package(Boost...) will reset
  # Boost_LIBRARIES.
  find_package(Boost QUIET COMPONENTS thread)
  set(ALL_BOOST_LIBS "${ALL_BOOST_LIBS};${Boost_LIBRARIES}")
  find_package(Boost QUIET COMPONENTS move)
  set(ALL_BOOST_LIBS "${ALL_BOOST_LIBS};${Boost_LIBRARIES}")
else()
  # If HPX was found, we use the Boost libraries found by HPX
  set(ALL_BOOST_LIBS ${HPX_LIBRARIES})
endif()

find_package(Boost QUIET COMPONENTS serialization)
set(Boost_SERIALIZATION_LIBRARIES "${Boost_LIBRARIES}")

find_package(Boost QUIET COMPONENTS mpi)
set(Boost_MPI_LIBRARIES "${Boost_LIBRARIES}")

# search for external LibFlatArray, but -- if none was found -- default to bundled version:
find_package(libflatarray QUIET)
if(NOT libflatarray_FOUND)
  add_subdirectory(lib/libflatarray)
  # add this include paths only to BUILD_INCLUDE_DIRECTORIES as must
  # not be exported during install in any case (the build tree must be
  # considered a temporary directory, and CMake would complain).
  list(APPEND BUILD_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/lib/libflatarray/include")
else()
  message(STATUS External LibFlatArray at ${libflatarray_INCLUDE_DIRS})
  # if LibFlatArray was found outside, we'll assume we're not using
  # the bundled version and we need export the relevant directory:
  list(APPEND INSTALL_INCLUDE_DIRECTORIES ${libflatarray_INCLUDE_DIRS})
endif()

# currently VisIt only supports Python 2.x. This (undocumented?)
# feature of CMake allows us to avoid Python 3.y.
set(PythonLibs_FIND_VERSION 2)
# setting this variable is required since CMake 3.0
set(PythonLibs_FIND_VERSION_MAJOR 2)

find_package(CUDA QUIET)
find_package(Doxygen)
find_package(OpenCL)
find_package(MPI)
find_package(OpenCV)
find_package(OpenMP)
find_package(PythonLibs)
find_package(Ruby)
find_package(SCOTCH)
find_package(Silo)
find_package(Qt5 COMPONENTS OpenGL Gui Core Widgets)
find_package(VisIt)

# if a package was not found it's safer to set the corresponding
# variable to false, this avoids undefined/unintended behavior in some
# cases.
if(NOT DEFINED Boost_SERIALIZATION_FOUND)
  set(Boost_SERIALIZATION_FOUND false)
endif()

if(NOT DEFINED Boost_MPI_FOUND)
  set(Boost_MPI_FOUND false)
endif()

if(NOT DEFINED Boost_MOVE_FOUND)
  set(Boost_MOVE_FOUND false)
endif()

if(NOT DEFINED CUDA_FOUND)
  set(CUDA_FOUND false)
endif()

if(NOT DEFINED DOXYGEN_FOUND)
  set(DOXYGEN_FOUND false)
endif()

if(NOT DEFINED HPX_FOUND)
  set(HPX_FOUND false)
endif()

if(NOT DEFINED OpenCV_FOUND)
  set(OpenCV_FOUND false)
endif()

if(NOT DEFINED OPENMP_FOUND)
  set(OPENMP_FOUND false)
endif()

if(NOT DEFINED PYTHONLIBS_FOUND)
  set(PYTHONLIBS_FOUND false)
endif()

if(NOT DEFINED Qt5_FOUND)
  set(Qt5_FOUND false)
endif()

if(NOT DEFINED RUBY_FOUND)
  set(RUBY_FOUND false)
endif()

if(NOT DEFINED SCOTCH_FOUND)
  set(SCOTCH_FOUND false)
endif()

if(NOT DEFINED Silo_FOUND)
  set(Silo_FOUND false)
endif()

#============= 2. DETECT DEFAULTS ====================================
set(RELEASE false)
lgd_detect_distro()

# A word on the various CMake compiler flag variables used inside LibGeoDecomp:
#
# DEFAULT_C_FLAGS, DEFAULT_CXX_FLAGS, NVCC_FLAGS  standard flags which may depend on the compiler,
#         |                 |             |       but not on 3rd party packages being installed.
#        _|_               _|_            |
#        \ /               \ /            |
#         Y                 Y             |
# LGD_ADDITIONAL_C/CXX_COMPILE_FLAGS      |       May be overridden by the user via CMake.
#         |                 |             |
#        _|_               _|_            |
#        \ /               \ /            |
#         Y                 Y             |
# LGD_AGGREGATED_C/CXX_FLAGS              |       Adds flags exported from packages used by LGD,
#         |                 |             |       exported to the parent scope of a CMake project
#        _|_               _|_            |       if LGD is nested.
#        \ /               \ /            |
#         Y                 Y             |
# CMAKE_C_FLAGS,    CMAKE_CXX_FLAGS       |       Default CMake variables
#         |                 |             |
#        _|_               _|_           _|_
#        \ /               \ /           \ /
#         Y                 Y             Y
#    C compiler      C++ compiler        nvcc
#                           |
#                          _|_
#                          \ /
#                           Y
#                     libgeodecomp_FLAGS          Exported via the CMake module
#
if (MSVC)
  # -Wall on Microsoft Visual C++ generates too much noise
  set(DEFAULT_C_FLAGS   "${DEFAULT_C_FLAGS}   /W4")
  set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /W4")
else()
  set(DEFAULT_C_FLAGS   "${DEFAULT_C_FLAGS}   -Wall")
  set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -Wall")
endif()

check_cxx_compiler_flag("-Wno-sign-conversion" SUPPORTS_WARN_NO_WSIGN)
if(SUPPORTS_WARN_NO_WSIGN)
  set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -Wno-sign-conversion")
endif()

check_cxx_compiler_flag("-Wnon-virtual-dtor" SUPPORTS_WARN_NON_VIRTUAL_DTOR)
if(SUPPORTS_WARN_NON_WSIGN)
  set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -Wnon-virtual-dtor")
endif()

check_cxx_compiler_flag("-fdiagnostics-color=always" SUPPORTS_COLOR)
if(SUPPORTS_COLOR)
  set(DEFAULT_C_FLAGS   "${DEFAULT_C_FLAGS}   -fdiagnostics-color=always")
  set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -fdiagnostics-color=always")
endif()

if(MSVC)
  set(SUPPORTS_CPP14 true)
else()
  check_cxx_compiler_flag("-std=c++14" SUPPORTS_CPP14)
  if(NOT SUPPORTS_CPP14)
    set(SUPPORTS_CPP14 false)
  endif()
endif()

check_cxx_compiler_flag("-march=native" SUPPORTS_MARCH_NATIVE)
if(SUPPORTS_MARCH_NATIVE AND NOT APPLE)
  set(DEFAULT_C_FLAGS   "${DEFAULT_C_FLAGS}   -march=native")
  set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -march=native")
endif()

# configure whether the dreaded typemap-generator will run:
set(DEFAULT_TYPEMAP_GENERATION false)
if((NOT RELEASE) AND DOXYGEN_FOUND AND RUBY_FOUND)
  set(DEFAULT_TYPEMAP_GENERATION true)
endif()

execute_process(
  COMMAND uname -m
  OUTPUT_VARIABLE MACHINE_ARCH
  ERROR_QUIET)

set(AMD64_LINUX false)
if (MACHINE_ARCH MATCHES "x86_64")
  set(AMD64_LINUX true)
endif()

#============= 3. CONFIGURABLE BUILD OPTIONS =========================
lgd_add_config_option(LGD_ADDITIONAL_C_COMPILE_FLAGS   "Add these flags when compiling C code."   "${DEFAULT_C_FLAGS}"   false)
lgd_add_config_option(LGD_ADDITIONAL_CXX_COMPILE_FLAGS "Add these flags when compiling C++ code." "${DEFAULT_CXX_FLAGS}" false)

lgd_add_config_option(CMAKE_BUILD_TYPE "Sets the compile/link options, e.g. Debug, Release... Refer to the cmake documentation for more details." "Release" false)

lgd_add_config_option(CMAKE_CXX_COMPILER "Select the C++ compiler" ${CMAKE_CXX_COMPILER} false)

lgd_add_config_option(CMAKE_INSTALL_PREFIX "Path for installation" "/usr/local" false)

# setting DEBUG_LEVEL to " 0" is a hack to ensure it's included in the config header, as just "0" would evaluate to false (see util.cmake)
lgd_add_config_option(DEBUG_LEVEL "Configure the amount/verbosity of debug output" " 0" true)

lgd_add_config_option(LIB_LINKAGE_TYPE "Controls which type of library to build. Suggested: SHARED on Linux (creates a shared object \"libgeodecomp.so\"), STATIC should work for builds on Cray and Windows." "SHARED" false)

lgd_add_config_option(LIB_DIR "Library subdirectory (e.g. lib, lib32, lib64), useful for multilib builds" "lib" false)

lgd_add_config_option(LIMIT_TESTS "Limit the directories of tests to run. Only tests matching the specified pattern will be run." false false)

lgd_add_config_option(UNITEXEC "May be used to specify a wrapper which then calls a unit test executable. Handy if for instance the unit tests shall be run on a remote machine." "" false)

lgd_add_config_option(WITH_BOOST_ASIO "Some Boost distributions ship too old versions of ASIO which are not compatible with LibGeoDecomp. Use this flag to switch ASIO-related code off." false true)

lgd_add_config_option(WITH_BOOST_MOVE "Enable/disable Boost.Move for move semantics (e.g. to avoid copies of vectors)." ${Boost_MOVE_FOUND} true)

lgd_add_config_option(WITH_BOOST_MPI "Enable/disable Boost.MPI related code." ${Boost_MPI_FOUND} true)

lgd_add_config_option(WITH_BOOST_SERIALIZATION "Explicitly disable Boost.Serialization, which isn't available on some machines (e.g. woody.rrze.uni-erlangen.de). Don't do this if you need the HPX backend." ${Boost_SERIALIZATION_FOUND} true)

lgd_add_config_option(WITH_BOOST_SHARED_PTR "Lets LibGeoDecomp use boost::shared_ptr instead of std::shared_ptr. Useful if your C++ compiler doesn't support the latter." false true)

lgd_add_config_option(WITH_CUDA "Enable modules which harness NVIDIA CUDA GPUs" ${CUDA_FOUND} true)

lgd_add_config_option(WITH_FORTRAN "Build Fortran examples/utilities, too" false true)

lgd_add_config_option(WITH_HPX "Build those modules which require HPX" ${HPX_FOUND} true)

lgd_add_config_option(WITH_INTRINSICS "Switch on/off the code parts which require SSE or AVX intrinsics" ${AMD64_LINUX} true)

lgd_add_config_option(WITH_LAX_VISIT_TESTS "Remove some of the stricter assertions from VisIt related unit tests -- VisIt sometimes produces erroneous results when running on a remote machine (e.g. for autobuilds)" false true)

lgd_add_config_option(WITH_LIBPTHREAD "Use this option to avoid linking against libpthread but instead adding -pthread to the compiler options (the former is required for CUDA when using nvcc, the latter for Android with gcc)." true false)

lgd_add_config_option(WITH_MPI "If set, all MPI related components will be built. This option is required for the MPI unit tests." ${MPI_FOUND} true)

lgd_add_config_option(WITH_OPENCL "Enable modules for delegating to OpenCL devices" false true)

lgd_add_config_option(WITH_OPENCV "Build those modules which require OpenCV" ${OpenCV_FOUND} false)

lgd_add_config_option(WITH_QT5 "Build example codes which rely on QT5 for the GUI" ${Qt5_FOUND} true)

lgd_add_config_option(WITH_SCOTCH "Enables LibGeoDecomp to use Scotch and PT-Scotch for domain decomposition." ${SCOTCH_FOUND} true)

lgd_add_config_option(WITH_SILO "Silo is a flexible output library developed by LLNL." ${Silo_FOUND} true)

lgd_add_config_option(WITH_THREADS "Lets you control whether we'll use threads (e.g. boost::thread)" ${OPENMP_FOUND} true)

lgd_add_config_option(WITH_TYPEMAPS "Controls whether the build system should regenerate typemaps.{h,cpp}. Requires Ruby and some Unix tools." ${DEFAULT_TYPEMAP_GENERATION} false)

lgd_add_config_option(WITH_VISIT "Activate code parts which use VisitWriter and SerialVisitWriter" ${VISIT_FOUND} true)

lgd_add_config_option(WITH_CPP14 "LibGeoDecomp is largely written in C++98, but some components use language features from a newer standard. This option will enable those." ${SUPPORTS_CPP14} true)

lgd_print_options()
lgd_dump_config("config.h")

#============= 4. RECURSE INTO SUBDIRECTORIES ========================
add_subdirectory(src)

#============= 5. EXPORT SETTINGS FOR NESTED USE =====================
if(lgd_has_parent)
  # export this for users who bundle a checkout of LibGeoDecomp in their own CMake source tree
  set(libgeodecomp_ALL_INCLUDE_DIRECTORIES ${LIBGEODECOMP_ALL_INCLUDE_DIRECTORIES} PARENT_SCOPE)
  set(libgeodecomp_FLAGS ${LGD_AGGREGATED_CXX_FLAGS} PARENT_SCOPE)
endif()
