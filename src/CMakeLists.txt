cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
project(LIBGEODECOMP)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(util.cmake)

#============= DETECT DEFAULTS ====================================
if(EXISTS "/etc/lsb-release")
  file(READ "/etc/lsb-release" lsb_release)
  string(REGEX MATCH "\".*\"" distro "${lsb_release}")
  string(REPLACE "\"" "" distro "${distro}")
endif()

if(NOT DEFINED distro)
  if(EXISTS "/etc/fedora-release")
    set(distro "Fedora")
  endif()
endif()

if(NOT DEFINED distro)
  if(EXISTS "/etc/debian_version")
    set(distro "Debian")
  endif()
endif()

message("distro: ${distro}")

find_package(Boost REQUIRED date_time filesystem system)
find_package(CUDA)
find_package(OpenCL)
find_package(MPI)
find_package(Qt4)

if(NOT DEFINED ${QT4_FOUND})
  set(QT4_FOUND false)
endif()

#============= CONFIGURABLE BUILD OPTIONS =========================
add_config_option(GENERATE_TYPEMAPS "Controls whether the build system should regenerate typemaps.{h,cpp}. Requires Ruby and some Unix tools." ON false)

add_config_option(LIMIT_TESTS "Limit the directories of tests to run. Only tests matching the specified pattern will be run." OFF false)

add_config_option(CMAKE_INSTALL_PREFIX "Path for installation" "/usr/local" false)

add_config_option(CMAKE_BUILD_TYPE "Sets the compile/link options, e.g. Debug, Release... Refer to the cmake documentation for more details." "Release" false)

add_config_option(ADDITIONAL_COMPILE_FLAGS "Add these flags when compiling." "-O2 -Wall -Werror -Wno-sign-compare" false)

add_config_option(LIB_LINKAGE_TYPE "Controls which type of library to build. Suggested: SHARED on Linux (creates a shared object \"libgeodecomp.so\"), STATIC should work for Windows builds" "SHARED" false)

add_config_option(FEATURE_CUDA "Enable modules which harness Nvidia CUDA GPUs" ${CUDA_FOUND} true)

add_config_option(FEATURE_MPI "If set, all MPI related components will be built. This option is required for the MPI unit tests." ${MPI_FOUND} true)

add_config_option(FEATURE_OPENCL "Enable modules for delegating to OpenCL devices" ${OPENCL_FOUND} true)

add_config_option(FEATURE_QT "Build example codes which rely on QT4 for the GUI" ${QT4_FOUND} true)

print_options()
dump_config("config.h")

#============= GENERAL CONFIGURATION ==============================
include_directories(${LIBGEODECOMP_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

# LIBDIRS is used to find source files and headers 
set(LIBDIRS io loadbalancer misc mpilayer parallelization parallelization/hiparsimulator parallelization/hiparsimulator/partitions)
# AUXDIRS lists auxiliary directories to be included in the main
# build. They may for instance include additionally libraries to be
# linked into the main lib.
set(AUXDIRS)
# set(AUXDIRS misc/testbed/cell/spustuff)
set(SUBDIRS examples io loadbalancer misc mpilayer parallelization testbed)

set(PACKAGE_NAME libgeodecomp)
set(CMAKE_C_FLAGS   "${ADDITIONAL_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${ADDITIONAL_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")

set(CXX_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/cxxtest")

set(PACKAGE_VERSION "0.1.0")
set(PACKAGE_VENDOR "Chair for Computer Science 3, FAU Erlangen, Germany")
set(PACKAGE_HOMEPAGE "http:\\\\www.libgeodecomp.org")
set(PACKAGE_EMAIL "users@libgeodecomp.org")

# import settings detected by packages to decouple from their
# interfaces, perform sanity checks to match selected options and
# detected environment...

if(FEATURE_CUDA AND NOT CUDA_FOUND)
  message("FEATURE_CUDA selected, but could not find the Nvidia CUDA toolkit.")
endif()

if(FEATURE_MPI)
  if(NOT MPI_FOUND)
    message("FEATURE_MPI selected, but could find no MPI implementation.")
  endif()

  if(MPI_FOUND)
    set(CMAKE_C_COMPILER   "${MPI_C_COMPILER}")
    set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
  endif()
endif()

if(FEATURE_OPENCL AND NOT OPENCL_FOUND)
  message("FEATURE_OPENCL selected, but could not find any OpenCL library.")
endif()

if(FEATURE_QT AND NOT QT4_FOUND)
  message("FEATURE_QT selected, but could not find QT4")
endif()

if(WIN32)
  set(LOCAL_LIBGEODECOMP_LINK_LIB local_libgeodecomp_lib)
else(WIN32)
  set(LOCAL_LIBGEODECOMP_LINK_LIB geodecomp)
endif(WIN32)

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

list(APPEND SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/mpilayer/typemaps.cpp")

set(ALL_SOURCES ${SOURCES})
set(ALL_HEADERS ${HEADERS})

add_library(geodecomp ${LIB_LINKAGE_TYPE} ${SOURCES})
target_link_libraries(geodecomp ${Boost_LIBRARIES})
if(FEATURE_OPENCL)
  target_link_libraries(geodecomp OpenCL)
endif(FEATURE_OPENCL)

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
    COMMAND cp -a "${CMAKE_BINARY_DIR}/last_cache" "${CMAKE_BINARY_DIR}/cache" 2>/dev/null || touch "${CMAKE_BINARY_DIR}/cache"
    COMMAND ruby "${CMAKE_CURRENT_SOURCE_DIR}/../tools/typemapgenerator/generate.rb" --cache "${CMAKE_BINARY_DIR}/cache" "${CMAKE_CURRENT_SOURCE_DIR}" "${CANDIDATES}"
    COMMAND cp -a "${CMAKE_BINARY_DIR}/cache" "${CMAKE_BINARY_DIR}/last_cache")
endif(GENERATE_TYPEMAPS)
