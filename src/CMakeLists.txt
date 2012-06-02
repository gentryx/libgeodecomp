cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
project(LIBGEODECOMP)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(util.cmake)
include(CheckCXXCompilerFlag)

#============= DETECT DEFAULTS ====================================
set(RELEASE false)

if(NOT DEFINED BOOST_ROOT)
  set(BOOST_ROOT "/opt/boost")
endif()

detect_distro()
find_package(Boost REQUIRED date_time filesystem system)
find_package(CUDA)
find_package(OpenCL)
find_package(MPI)
find_package(OpenCV)
find_package(Ruby)
find_package(Qt4)

if(NOT DEFINED QT4_FOUND)
  set(QT4_FOUND false)
endif()

if(NOT DEFINED RUBY_FOUND)
  set(RUBY_FOUND false)
endif()

if(NOT DEFINED CUDA_FOUND)
  set(CUDA_FOUND false)
endif()

if(NOT DEFINED OpenCV_FOUND)
  set(OpenCV_FOUND false)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT DEFINED ADD_BOOST_INCLUDE_PATH)
  if (distro MATCHES "Gentoo")
    set(ADD_BOOST_INCLUDE_PATH false)
  else()
    set(ADD_BOOST_INCLUDE_PATH true)
  endif()
endif()

set(DEFAULT_FLAGS "-O2 -Wall -Wno-sign-compare")
set(DEFAULT_TYPEMAP_GENERATION false)

if(NOT RELEASE)
  # Boost is creating too many warnings, -Werror is just too much
  # set(DEFAULT_FLAGS "${DEFAULT_FLAGS} -Werror")
  set(DEFAULT_TYPEMAP_GENERATION ${RUBY_FOUND})
endif()

execute_process(
  COMMAND uname -m
  OUTPUT_VARIABLE MACHINE_ARCH
  ERROR_QUIET)

CHECK_CXX_COMPILER_FLAG("-march=native" SUPPORTS_MARCH_NATIVE)
if(SUPPORTS_MARCH_NATIVE)
  set(DEFAULT_FLAGS "${DEFAULT_FLAGS} -march=native")
endif()

#============= CONFIGURABLE BUILD OPTIONS =========================
add_config_option(GENERATE_TYPEMAPS "Controls whether the build system should regenerate typemaps.{h,cpp}. Requires Ruby and some Unix tools." ${DEFAULT_TYPEMAP_GENERATION} false)

add_config_option(LIMIT_TESTS "Limit the directories of tests to run. Only tests matching the specified pattern will be run." OFF false)

add_config_option(CMAKE_INSTALL_PREFIX "Path for installation" "/usr/local" false)

add_config_option(CMAKE_BUILD_TYPE "Sets the compile/link options, e.g. Debug, Release... Refer to the cmake documentation for more details." "Release" false)

add_config_option(ADDITIONAL_COMPILE_FLAGS "Add these flags when compiling." "${DEFAULT_FLAGS}" false)

add_config_option(LIB_LINKAGE_TYPE "Controls which type of library to build. Suggested: SHARED on Linux (creates a shared object \"libgeodecomp.so\"), STATIC should work for Windows builds" "SHARED" false)

add_config_option(FEATURE_CUDA "Enable modules which harness Nvidia CUDA GPUs" ${CUDA_FOUND} true)

add_config_option(FEATURE_MPI "If set, all MPI related components will be built. This option is required for the MPI unit tests." ${MPI_FOUND} true)

add_config_option(FEATURE_OPENCL "Enable modules for delegating to OpenCL devices" ${OPENCL_FOUND} true)

add_config_option(FEATURE_QT "Build example codes which rely on QT4 for the GUI" ${QT4_FOUND} true)

add_config_option(FEATURE_OPENCV "Build those modules which require OpenCV" ${OpenCV_FOUND} false)

add_config_option(CMAKE_CXX_COMPILER "Select the C++ compiler" ${CMAKE_CXX_COMPILER} false)

print_options()
dump_config("config.h")

#============= GENERAL CONFIGURATION ==============================

set(PACKAGE_NAME libgeodecomp)
set(CMAKE_C_FLAGS   "${ADDITIONAL_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${ADDITIONAL_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")

set(PACKAGE_VERSION "0.0.1")
set(PACKAGE_VENDOR "Chair for Computer Science 3, FAU Erlangen, Germany")
set(PACKAGE_HOMEPAGE "http:\\\\www.libgeodecomp.org")
set(PACKAGE_EMAIL "users@libgeodecomp.org")

# import settings detected by packages to decouple from their
# interfaces, perform sanity checks to match selected options and
# detected environment...

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(FEATURE_MPI)
  add_definitions(${MPI_COMPILE_FLAGS})
  include_directories(${MPI_INCLUDE_PATH})
endif()
if(ADD_BOOST_INCLUDE_PATH)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

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

if(WIN32)
  set(LOCAL_LIBGEODECOMP_LINK_LIB local_libgeodecomp_lib)
else(WIN32)
  set(LOCAL_LIBGEODECOMP_LINK_LIB geodecomp)
endif(WIN32)

# LIBDIRS is used to find source files and headers 
set(LIBDIRS io loadbalancer misc mpilayer parallelization parallelization/hiparsimulator parallelization/hiparsimulator/partitions)
# AUXDIRS lists auxiliary directories to be included in the main
# build. They may for instance include additionally libraries to be
# linked into the main lib.
set(AUXDIRS)
# set(AUXDIRS misc/testbed/cell/spustuff)
set(SUBDIRS examples io loadbalancer misc mpilayer parallelization testbed)

set(CXX_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/cxxtest")

generate_sourcelists("./")

#============= TOP-LEVEL TARGETS ==================================
add_custom_target(test echo "Tests passed.")
add_custom_target(code_generation ALL echo "Code generation done.")

#============= INSTALLER STUFF ====================================
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
  set(CPACK_PACKAGE_ICON             "${CMake_SOURCE_DIR}/..\\libgeodecomp_icon.png")
  set(CPACK_NSIS_INSTALLED_ICON_NAME "${CMake_SOURCE_DIR}/..\\libgeodecomp_icon.png")
  # further options:
  #else(WIN32 AND NOT UNIX)
  #  set(CPACK_STRIP_FILES "bin/MyExecutable")
  #  set(CPACK_SOURCE_STRIP_FILES "")
  #  set(CPACK_PACKAGE_EXECUTABLES "helloworld" "My Hello World")
endif(WIN32 AND NOT UNIX)

include(CPack)

#============= RECURSE SUBDIRS ====================================
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

#============= ADD THE MAIN LIBRARY ===============================
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
target_link_libraries(geodecomp ${Boost_LIBRARIES})
if(FEATURE_MPI)
  target_link_libraries(geodecomp ${MPI_LIBRARIES})
  set_property(TARGET geodecomp PROPERTY LINK_FLAGS ${MPI_LINK_FLAGS})
endif()
if(FEATURE_OPENCL)
  target_link_libraries(geodecomp OpenCL)
endif()

#============= INSTALLER CONFIG ===================================
install(TARGETS geodecomp DESTINATION lib)
install(FILES config.h DESTINATION include/${PACKAGE_NAME})

# install all headers
foreach(dir ${LIBDIRS})
  set(HEADERS "")
  set(RELATIVE_PATH ${dir}/)
  include(${dir}/auto.cmake)
  install(FILES ${HEADERS} DESTINATION include/${PACKAGE_NAME}/${dir})
endforeach(dir)

#============= TYPEMAPS GENERATION ================================
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
