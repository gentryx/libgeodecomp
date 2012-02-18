cmake_minimum_required(VERSION 2.6)
project(LIBGEODECOMP)
include(conf.cmake)

include_directories(${LIBGEODECOMP_SOURCE_DIR})

if(WIN32)
  add_library(local_libgeodecomp_lib STATIC IMPORTED)
  set_property(TARGET local_libgeodecomp_lib PROPERTY IMPORTED_LOCATION ${LIBGEODECOMP_BINARY_DIR}/libgeodecomp/geodecomp.lib)
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

# link all sources 
set(SOURCES "")
foreach(dir ${LIBDIRS})
  set(RELATIVE_PATH ${dir}/)
  include(${dir}/auto.cmake)
endforeach(dir)

add_library(geodecomp ${LIB_LINKAGE_TYPE} ${SOURCES})
target_link_libraries(geodecomp boost_date_time${BOOST_LINK_SUFFIX})
target_link_libraries(geodecomp boost_filesystem${BOOST_LINK_SUFFIX})
target_link_libraries(geodecomp boost_system${BOOST_LINK_SUFFIX})
if (LIBGEODECOMP_FEATURE_OPENCL)
  target_link_libraries(geodecomp OpenCL)
endif (LIBGEODECOMP_FEATURE_OPENCL)
install(TARGETS geodecomp DESTINATION lib)

# install all headers
foreach(dir ${LIBDIRS})
  set(HEADERS "")
  set(RELATIVE_PATH ${dir}/)
  include(${dir}/auto.cmake)
  install(FILES ${HEADERS} DESTINATION include/${PACKAGE_NAME}/${dir})
endforeach(dir)

set(HEADERS "")
set(SOURCES "")

# recurse subdirs
foreach(dir ${SUBDIRS})
  add_subdirectory(${dir})
endforeach(dir)

foreach(dir ${AUXDIRS})
  set(HEADERS "")
  set(SOURCES "")
  include(${dir}/aux.cmake)
endforeach(dir)
