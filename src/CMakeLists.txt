cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
project(LIBGEODECOMP)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(util.cmake)
include(CheckCXXCompilerFlag)

#============= 1. FIND PACKAGES WE DEPEND ON =========================
if(NOT DEFINED BOOST_ROOT)
  # deduce Boost location from environment (e.g. on woody.rrze.uni-erlangen.de)
  set(BOOST_ENV_LIBDIR "$ENV{BOOST_LIBDIR}")
  if(BOOST_ENV_LIBDIR)
    set(Boost_NO_SYSTEM_PATHS true)
    set(BOOST_ROOT "$ENV{BOOST_LIBDIR}/../")
    set(Boost_LIBRARY_DIRS "$ENV{BOOST_LIBDIR}")
    set(Boost_INCLUDE_DIR  "$ENV{BOOST_INCDIR}")
  endif()
endif()

# mandatory Boost libs go into the first line, optional ones into
# their own call. This avoids leaving out any available lib during
# link time. The reason for this ugly multi-stage discovery is the
# following: since CMAKE 2.8.11 each call to find_package(Boost...)
# will reset Boost_LIBRARIES.
find_package(Boost REQUIRED COMPONENTS date_time filesystem system)
set(ALL_BOOST_LIBS "${Boost_LIBRARIES}")
find_package(Boost COMPONENTS mpi)
set(ALL_BOOST_LIBS "${ALL_BOOST_LIBS};${Boost_LIBRARIES}")
find_package(Boost COMPONENTS serialization)
set(ALL_BOOST_LIBS "${ALL_BOOST_LIBS};${Boost_LIBRARIES}")
find_package(Boost COMPONENTS thread)
set(ALL_BOOST_LIBS "${ALL_BOOST_LIBS};${Boost_LIBRARIES}")
find_package(Boost COMPONENTS move)
set(ALL_BOOST_LIBS "${ALL_BOOST_LIBS};${Boost_LIBRARIES}")

find_package(CUDA)
find_package(OpenCL)
find_package(MPI)
find_package(OpenCV)
find_package(Ruby)
find_package(Qt4 COMPONENTS QtOpenGL QtGui QtCore)
find_package(VisIt)

# find HPX
if(NOT HPX_ROOT)
  if(NOT $ENV{HPX_ROOT} STREQUAL "")
    set(HPX_ROOT $ENV{HPX_ROOT})
  endif()
endif()

if(HPX_ROOT)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${HPX_ROOT}/share/cmake-2.8/Modules)
  find_package(HPX)
endif()

if(HPX_FOUND)
  set(DEFAULT_FLAGS ${HPX_CXX_COMPILE_FLAGS})
  include_directories(${HPX_INCLUDE_DIR})
  link_directories(${HPX_LIBRARY_DIR})
  add_definitions(-DHPX_LIMIT=10)
  set(HPX_FOUND true)
else()
  set(HPX_FOUND false)
endif()

# if a package was not found it's safer to set the corresponding
# variable to false, this avoids undefined/unintended behavior in some
# cases.
if(NOT DEFINED QT4_FOUND)
  set(QT4_FOUND false)
endif()

if(NOT DEFINED RUBY_FOUND)
  set(RUBY_FOUND false)
endif()

if(NOT DEFINED CUDA_FOUND)
  set(CUDA_FOUND false)
endif()

if(NOT DEFINED HPX_FOUND)
  set(HPX_FOUND false)
endif()

if(NOT DEFINED OpenCV_FOUND)
  set(OpenCV_FOUND false)
endif()

if(NOT DEFINED Boost_SERIALIZATION_FOUND)
  set(Boost_SERIALIZATION_FOUND false)
endif()

if(NOT DEFINED Boost_MPI_FOUND)
  set(Boost_MPI_FOUND false)
endif()

if(NOT DEFINED Boost_MOVE_FOUND)
  set(Boost_MOVE_FOUND false)
endif()

#============= 2. DETECT DEFAULTS ====================================
set(RELEASE false)
detect_distro()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# compute default flags:
set(DEFAULT_FLAGS "${DEFAULT_FLAGS} -Wall -Wno-sign-promo")

check_cxx_compiler_flag("-march=native" SUPPORTS_MARCH_NATIVE)
if(SUPPORTS_MARCH_NATIVE)
  set(DEFAULT_FLAGS "${DEFAULT_FLAGS} -march=native")
endif()

if(OPENCL_FOUND)
  check_cxx_compiler_flag("-std=c++11" SUPPORTS_CXX_11)
  if(SUPPORTS_CXX_11)
    set(DEFAULT_FLAGS "${DEFAULT_FLAGS} -std=c++11")
  endif()
endif()

# configure whether the dreaded typemap-generator will run:
set(DEFAULT_TYPEMAP_GENERATION false)
if(NOT RELEASE)
  set(DEFAULT_TYPEMAP_GENERATION ${RUBY_FOUND})
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
add_config_option(GENERATE_TYPEMAPS "Controls whether the build system should regenerate typemaps.{h,cpp}. Requires Ruby and some Unix tools." ${DEFAULT_TYPEMAP_GENERATION} false)

add_config_option(LIMIT_TESTS "Limit the directories of tests to run. Only tests matching the specified pattern will be run." OFF false)

add_config_option(CMAKE_INSTALL_PREFIX "Path for installation" "/usr/local" false)

add_config_option(CMAKE_BUILD_TYPE "Sets the compile/link options, e.g. Debug, Release... Refer to the cmake documentation for more details." "Release" false)

add_config_option(ADDITIONAL_COMPILE_FLAGS "Add these flags when compiling." "${DEFAULT_FLAGS}" false)

add_config_option(LIB_LINKAGE_TYPE "Controls which type of library to build. Suggested: SHARED on Linux (creates a shared object \"libgeodecomp.so\"), STATIC should work for builds on Cray and Windows." "SHARED" false)

add_config_option(FEATURE_CUDA "Enable modules which harness Nvidia CUDA GPUs" ${CUDA_FOUND} true)

add_config_option(FEATURE_MPI "If set, all MPI related components will be built. This option is required for the MPI unit tests." ${MPI_FOUND} true)

add_config_option(FEATURE_BOOST_MPI "Enable/disable Boost.MPI related code." ${Boost_MPI_FOUND} true)

add_config_option(FEATURE_BOOST_MOVE "Enable/disable Boost.Move for move semantics (e.g. to avoid copies of vectors)." ${Boost_MOVE_FOUND} true)

add_config_option(FEATURE_BOOST_SERIALIZATION "Explicitly disable Boost.Serialization, which isn't available on some machines (e.g. woody.rrze.uni-erlangen.de). Don't do this if you need the HPX backend." ${Boost_SERIALIZATION_FOUND} true)

add_config_option(FEATURE_OPENCL "Enable modules for delegating to OpenCL devices" false true)

add_config_option(FEATURE_QT "Build example codes which rely on QT4 for the GUI" ${QT4_FOUND} true)

add_config_option(FEATURE_OPENCV "Build those modules which require OpenCV" ${OpenCV_FOUND} false)

add_config_option(FEATURE_INTRINSICS "Switch on/off the code parts which require SSE or AVX intrinsics" ${AMD64_LINUX} true)

add_config_option(FEATURE_HPX "Build those modules which require HPX" ${HPX_FOUND} true)

add_config_option(FEATURE_VISIT "Activate code parts which use VisitWriter and SerialVisitWriter" ${VISIT_FOUND} true)

add_config_option(FEATURE_THREADS "Lets you control whether we'll use threads (e.g. boost::thread)" true true)

# setting DEBUG_LEVEL to "0 " is a hack to ensure it's included in the config header, as just "0" would evaluate to false (see util.cmake)
add_config_option(DEBUG_LEVEL "Configure the amount/verbosity of debug output" "0 " true)

add_config_option(CAKE_CXX_COMPILER "Select the C++ compiler" ${CMAKE_CXX_COMPILER} false)

add_config_option(UNITEXEC "May be used to specify a wrapper which then calls a unit test executable. Handy if for instance the unit tests shall be run on a remote machine." "" false)

print_options()
dump_config("config.h")

#============= 4. GENERAL CONFIGURATION ==============================
set(PACKAGE_NAME libgeodecomp)
set(CMAKE_C_FLAGS   "${ADDITIONAL_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${ADDITIONAL_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")

set(PACKAGE_VERSION "0.2.0")
set(PACKAGE_VENDOR "Chair for Computer Science 3, FAU Erlangen, Germany")
set(PACKAGE_HOMEPAGE "http:\\\\www.libgeodecomp.org")
set(PACKAGE_EMAIL "users@libgeodecomp.org")

# LIBDIRS is used to find source files and headers
set(LIBDIRS io loadbalancer misc mpilayer parallelization parallelization/hiparsimulator parallelization/hiparsimulator/partitions parallelization/hpxsimulator)
# AUXDIRS lists auxiliary directories to be included in the main
# build. They may for instance include additionally libraries to be
# linked into the main lib.
set(AUXDIRS)
# set(AUXDIRS misc/testbed/cell/spustuff)
set(SUBDIRS examples io loadbalancer misc mpilayer parallelization testbed)

set(CXX_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/cxxtest")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
generate_sourcelists("./")

#============= 5. SANITY CHECKS ======================================
# import settings detected by packages to decouple from their
# interfaces, perform sanity checks to match selected options and
# detected environment...
if(FEATURE_MPI)
  add_definitions(${MPI_COMPILE_FLAGS})
  include_directories(${MPI_INCLUDE_PATH})
endif()

include_directories(${Boost_INCLUDE_DIRS})

if(FEATURE_OPENCV AND NOT OpenCV_FOUND)
  message(FATAL_ERROR "FEATURE_OPENCV selected, but could not find OpenCV.")
endif()

if(FEATURE_CUDA AND NOT CUDA_FOUND)
  message(FATAL_ERROR "FEATURE_CUDA selected, but could not find the Nvidia CUDA toolkit.")
endif()

if(FEATURE_MPI)
  if(NOT MPI_FOUND)
    message(FATAL_ERROR "FEATURE_MPI selected, but could find no MPI implementation.")
  endif()

  if(NOT MPIEXEC)
    message(FATAL_ERROR "FEATURE_MPI selected, but mpiexec not found.")
  endif()
endif()

if(FEATURE_OPENCL AND NOT OPENCL_FOUND)
  message(FATAL_ERROR "FEATURE_OPENCL selected, but could not find any OpenCL library.")
endif()

if(FEATURE_QT AND NOT QT4_FOUND)
  message(FATAL_ERROR "FEATURE_QT selected, but could not find QT4")
endif()

if(FEATURE_HPX AND NOT HPX_FOUND)
  message(FATAL_ERROR "FEATURE_HPX selected but could not find HPX. Specify HPX_ROOT to point to your HPX installation.")
endif()

if(FEATURE_HPX AND NOT Boost_SERIALIZATION_FOUND)
  message(FATAL_ERROR "FEATURE_HPX selected but Boost.Serialization is missing. Is your Boost installation incomplete?")
endif()

if (FEATURE_BOOST_MPI AND NOT Boost_MPI_FOUND)
  message(FATAL_ERROR "FEATURE_BOOST_MPI selected, but Boost.MPI library not found")
endif()

if (FEATURE_BOOST_MOVE AND NOT Boost_MOVE_FOUND)
  message(FATAL_ERROR "FEATURE_BOOST_MOVE selected, but Boost.Move library not found")
endif()

if (FEATURE_BOOST_SERIALIZATION AND NOT Boost_SERIALIZATION_FOUND)
  message(FATAL_ERROR "FEATURE_BOOST_SERIALIZATION selected, but Boost.Serialization library not found")
endif()

if(FEATURE_VISIT)
  include_directories(${VisIt_INCLUDE_DIRS})
endif()

if(WIN32)
  set(LOCAL_LIBGEODECOMP_LINK_LIB local_libgeodecomp_lib)
else(WIN32)
  set(LOCAL_LIBGEODECOMP_LINK_LIB geodecomp)
endif(WIN32)

if(FEATURE_HPX)
  set(HPX_LIB_SUFFIX "")
  if("${HPX_BUILD_TYPE}" STREQUAL "Debug")
    set(HPX_LIB_SUFFIX ${HPX_DEBUG_POSTFIX})
  endif()
endif()

if(FEATURE_QT)
  include(${QT_USE_FILE})
  add_definitions(-DQT_NO_KEYWORDS)
endif()

#============= 6. TOP-LEVEL TARGETS ==================================
add_custom_target(test echo "Tests passed.")
add_custom_target(code_generation ALL echo "Code generation done.")
add_custom_target(doc cd "${CMAKE_CURRENT_SOURCE_DIR}" && doxygen doxygen.conf)

#============= 7. INSTALLER STUFF ====================================
set(CPACK_PACKAGE_NAME ${PACKAGE_NAME})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION ${PACKAGE_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${PACKAGE_NAME})

# will be shown e.g. in windows' control center package info
set(CPACK_PACKAGE_VENDOR ${PACKAGE_VENDOR})
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../README")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${PACKAGE_NAME})

if(WIN32 AND NOT UNIX)
  # unused in current nsis versions, fix pending (http://www.cmake.org/Bug/print_bug_page.php?bug_id=8682)
  set(CPACK_NSIS_DISPLAY_NAME ${PACKAGE_NAME})

  # name to show in software tab of control center
  set(CPACK_NSIS_DISPLAY_NAME ${PACKAGE_NAME})
  # will be shown e.g. in windows' control center package info
  set(CPACK_NSIS_HELP_LINK ${PACKAGE_HOMEPAGE})
  # will be shown e.g. in windows' control center package info
  set(CPACK_NSIS_URL_INFO_ABOUT ${PACKAGE_HOMEPAGE})
  # will be shown e.g. in windows' control center package info
  set(CPACK_NSIS_CONTACT ${PACKAGE_EMAIL})
  set(CPACK_NSIS_MODIFY_PATH ON)

  # There is a bug in NSI that does not handle full unix paths properly. Make
  # sure there is at least one set of four (4) backlasshes.
  set(CPACK_PACKAGE_ICON             "${CMAKE_SOURCE_DIR}/..\\libgeodecomp_icon.png")
  set(CPACK_NSIS_INSTALLED_ICON_NAME "${CMAKE_SOURCE_DIR}/..\\libgeodecomp_icon.png")
  # further options:
  #else(WIN32 AND NOT UNIX)
  #  set(CPACK_STRIP_FILES "bin/MyExecutable")
  #  set(CPACK_SOURCE_STRIP_FILES "")
  #  set(CPACK_PACKAGE_EXECUTABLES "helloworld" "My Hello World")
endif(WIN32 AND NOT UNIX)

include(CPack)

#============= 8. RECURSE SUBDIRS ====================================
set(HEADERS "")
set(SOURCES "")

foreach(dir ${SUBDIRS})
  add_subdirectory(${dir})
endforeach(dir)

foreach(dir ${AUXDIRS})
  set(HEADERS "")
  set(SOURCES "")
  include(${dir}/aux.cmake)
endforeach(dir)

#============= 9. ADD THE MAIN LIBRARY ===============================
if(WIN32)
  add_library(local_libgeodecomp_lib STATIC IMPORTED)
  set_property(TARGET local_libgeodecomp_lib PROPERTY IMPORTED_LOCATION ${LIBGEODECOMP_BINARY_DIR}/libgeodecomp/geodecomp.lib)
endif(WIN32)

# link all sources
set(SOURCES "")
foreach(dir ${LIBDIRS})
  set(RELATIVE_PATH ${dir}/)
  include(${dir}/auto.cmake)
endforeach(dir)

set(ALL_SOURCES ${SOURCES})
set(ALL_HEADERS ${HEADERS})

add_library(geodecomp ${LIB_LINKAGE_TYPE} ${SOURCES})

# link 3rd party libraries to main lib and add them to pkg-config file down below
set(LIBGEODECOMP_LINK_LIBRARIES ${ALL_BOOST_LIBS})
if(FEATURE_HPX)
  set(LIBGEODECOMP_LINK_LIBRARIES
    ${LIBGEODECOMP_LINK_LIBRARIES}
    hpx${HPX_LIB_SUFFIX}
    hpx_serialization${HPX_LIB_SUFFIX}
    hpx_init${HPX_LIB_SUFFIX}
    ${HPX_LIBRARIES}
  )
endif()

if(FEATURE_QT)
  set(LIBGEODECOMP_LINK_LIBRARIES
    ${LIBGEODECOMP_LINK_LIBRARIES}
    ${QT_LIBRARIES})
endif()

if(FEATURE_THREADS)
  # some systems (e.g. my devel notebook) require manually linking pthreads for OpenMP...
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    set(LIBGEODECOMP_LINK_LIBRARIES ${LIBGEODECOMP_LINK_LIBRARIES} pthread)
  endif()
endif()

if(FEATURE_MPI)
  set(LIBGEODECOMP_LINK_LIBRARIES ${LIBGEODECOMP_LINK_LIBRARIES} ${MPI_LIBRARIES})
  set_property(TARGET geodecomp PROPERTY LINK_FLAGS ${MPI_LINK_FLAGS})
endif()

if(FEATURE_OPENCL)
  set(LIBGEODECOMP_LINK_LIBRARIES ${LIBGEODECOMP_LINK_LIBRARIES} OpenCL)
endif()

if(FEATURE_VISIT)
  set(LIBGEODECOMP_LINK_LIBRARIES ${LIBGEODECOMP_LINK_LIBRARIES} ${VisIt_LIBRARIES} dl)
endif()

target_link_libraries(geodecomp ${LIBGEODECOMP_LINK_LIBRARIES})

#============= 10. INSTALLER CONFIG ==================================
install(TARGETS geodecomp DESTINATION lib)
install(FILES config.h DESTINATION include/${PACKAGE_NAME})
install(FILES libgeodecomp.h DESTINATION include)

# install all headers
foreach(dir ${LIBDIRS})
  set(HEADERS "")
  set(RELATIVE_PATH ${dir}/)
  include(${dir}/auto.cmake)
  install(FILES ${HEADERS} DESTINATION include/${PACKAGE_NAME}/${dir})
endforeach(dir)
# install libflatarray, too
install(FILES ../lib/libflatarray/src/flat_array.hpp DESTINATION include/libflatarray)

# link against the libs we need
set(INSTALLER_LINK_FLAGS "-L\${libdir} -lgeodecomp ")
foreach(lib ${LIBGEODECOMP_LINK_LIBRARIES} /usr/lib/liblibkremotecontrol.so)
  get_filename_component(path "${lib}" PATH)
  get_filename_component(name "${lib}" NAME_WE)

  if(name MATCHES ^lib)
    # this dirty trick is required to prevent REGEX REPLACE matching
    # every occurrence of "lib" ("^lib" will sadly match all, too, if
    # we're going to remove them)
    set(name "_dummy_${name}")
    string(REGEX REPLACE "_dummy_lib" "" name "${name}")
  endif()

  if (path)
    set(INSTALLER_LINK_FLAGS "${INSTALLER_LINK_FLAGS} -L${path}")
  endif()
  set(INSTALLER_LINK_FLAGS "${INSTALLER_LINK_FLAGS} -l${name}")
endforeach()

file(WRITE "${CMAKE_BINARY_DIR}/${PACKAGE_NAME}.pc" "prefix=${CMAKE_INSTALL_PREFIX}
libdir=${CMAKE_INSTALL_PREFIX}/lib
includedir=${CMAKE_INSTALL_PREFIX}/include/libgeodecomp

Name: LibGeoDecomp
Description: High performance stencil code library
Version: ${PACKAGE_VERSION}
Libs: ${INSTALLER_LINK_FLAGS}
Cflags: -I\${includedir}
")
install(FILES "${CMAKE_BINARY_DIR}/${PACKAGE_NAME}.pc" DESTINATION "lib/pkgconfig")

#============= 11. TYPEMAPS GENERATION ===============================
if (GENERATE_TYPEMAPS)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/mpilayer/typemaps.h" "${CMAKE_CURRENT_SOURCE_DIR}/mpilayer/typemaps.cpp"
    COMMAND cd "${CMAKE_CURRENT_SOURCE_DIR}" && doxygen doxygen.conf
    COMMAND ruby "${CMAKE_CURRENT_SOURCE_DIR}/../tools/typemapgenerator/generate.rb" --namespace LibGeoDecomp --header-fix "^.+/src:libgeodecomp" --macro-guard LIBGEODECOMP_FEATURE_MPI "${CMAKE_CURRENT_SOURCE_DIR}/../doc/xml" "${CMAKE_CURRENT_SOURCE_DIR}/mpilayer"
    DEPENDS "${CMAKE_BINARY_DIR}/cache")

  list(REMOVE_ITEM ALL_HEADERS "mpilayer/typemaps.h")
  set(CANDIDATES)
  foreach(i ${ALL_HEADERS})
    set(CANDIDATES "${i}:${CANDIDATES}")
  endforeach(i)

  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/cache"
    COMMAND cp -a "${CMAKE_BINARY_DIR}/last_cache" "${CMAKE_BINARY_DIR}/cache" 2>/dev/null || echo "still ok >/dev/null"
    COMMAND ruby "${CMAKE_CURRENT_SOURCE_DIR}/../tools/typemapgenerator/generate.rb" --cache "${CMAKE_BINARY_DIR}/cache" "${CMAKE_CURRENT_SOURCE_DIR}" "${CANDIDATES}"
    COMMAND cp -a "${CMAKE_BINARY_DIR}/cache" "${CMAKE_BINARY_DIR}/last_cache")
endif(GENERATE_TYPEMAPS)
