# - Tries to find VisIt
# You may define VisIt_ROOT to hint at the install location. Once done this will define
#  VISIT_FOUND - VisIt is installed
#  VisIt_INCLUDE_DIRS - Directories where VisIt headers can be found
#  VisIt_LIBRARIES - Libraries required by VisIt headers

set(VisIt_VERSIONS 2.4.0 2.4.1 2.4.2 2.5.0 2.5.1 2.5.2 2.6.0 2.6.1 2.6.2 2.7.0 2.7.1 2.7.2 2.7.3 2.8.0 2.8.1 2.8.2 2.9.0 2.10.0 2.10.1 2.10.2)
set(VisIt_ARCHITECTURES linux-x86_64)
set(VisIt_CANDIDATES /opt/visit /opt/visit/visit ${VisIt_ROOT})
set(VisIt_SUFFIXES)
foreach(version ${VisIt_VERSIONS})
  foreach(arch ${VisIt_ARCHITECTURES})
    list(APPEND VisIt_SUFFIXES ${version}/${arch}/libsim/V2)
  endforeach()
endforeach()

set(VisIt_INC_SUFFIXES)
set(VisIt_LIB_SUFFIXES)
foreach(suffix ${VisIt_SUFFIXES})
  list(APPEND VisIt_INC_SUFFIXES ${suffix}/include)
  list(APPEND VisIt_LIB_SUFFIXES ${suffix}/lib)
endforeach()

find_path(VisIt_INCLUDE_DIRS VisItControlInterface_V2.h HINTS ${VisIt_CANDIDATES} PATH_SUFFIXES ${VisIt_INC_SUFFIXES} ENV CPLUS_INCLUDE_PATH)
find_library(VisIt_LIBRARIES simV2  HINTS ${VisIt_CANDIDATES} PATH_SUFFIXES ${VisIt_LIB_SUFFIXES} ENV LIBRARY_PATH LD_LIBRARY_PATH)

find_package_handle_standard_args(VisIt DEFAULT_MSG VisIt_INCLUDE_DIRS VisIt_LIBRARIES)

set(VISIT_FOUND ${VISIT_FOUND} CACHE BOOL "")
set(VisIt_INCLUDE_DIRS ${VisIt_INCLUDE_DIRS} CACHE FILEPATH "")
set(VisIt_LIBRARIES ${VisIt_LIBRARIES} CACHE FILEPATH "")

mark_as_advanced(VISIT_FOUND VisIt_INCLUDE_DIRS VisIt_LIBRARIES)
