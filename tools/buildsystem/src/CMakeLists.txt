cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR)
project(LIBGEODECOMP)
include(FeatureSummary)
include(conf.cmake)
include(util.cmake)

# configurable build options
option(LIMIT_TESTS "Limit the directories of tests to run." OFF)
add_feature_info(LimitTests LIMIT_TESTS "Only tests matching the specified pattern will be run.")
feature_summary(WHAT ALL)

# top-level
add_custom_target(test echo "Tests passed.")
add_custom_target(code_generation ALL echo "Code generation done.")

include_directories(${LIBGEODECOMP_SOURCE_DIR})

# LIBDIRS is used to find source files and headers 
set(LIBDIRS io loadbalancer misc mpilayer parallelization parallelization/hiparsimulator parallelization/hiparsimulator/partitions)
# AUXDIRS lists auxiliary directories to be included in the main
# build. They may for instance include additionally libraries to be
# linked into the main lib.
set(AUXDIRS)
# set(AUXDIRS misc/testbed/cell/spustuff)
set(SUBDIRS examples io loadbalancer misc mpilayer parallelization testbed)

generate_sourcelists("./")

if(WIN32)
  add_library(local_libgeodecomp_lib STATIC IMPORTED)
  set_property(TARGET local_libgeodecomp_lib PROPERTY IMPORTED_LOCATION ${LIBGEODECOMP_BINARY_DIR}/libgeodecomp/geodecomp.lib)
  set(LOCAL_LIBGEODECOMP_LINK_LIB local_libgeodecomp_lib)
else(WIN32)
  set(LOCAL_LIBGEODECOMP_LINK_LIB geodecomp)
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
target_link_libraries(geodecomp boost_date_time${BOOST_LINK_SUFFIX})
target_link_libraries(geodecomp boost_filesystem${BOOST_LINK_SUFFIX})
target_link_libraries(geodecomp boost_system${BOOST_LINK_SUFFIX})
if (LIBGEODECOMP_FEATURE_OPENCL)
  target_link_libraries(geodecomp OpenCL)
endif (LIBGEODECOMP_FEATURE_OPENCL)
install(TARGETS geodecomp DESTINATION lib)
install(FILES config.h DESTINATION include/${PACKAGE_NAME})

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

# typemaps generation 
# fixme: make generation a configurable option (default: off for tarballs, on for checkouts)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/mpilayer/typemaps.h" "${CMAKE_CURRENT_SOURCE_DIR}/mpilayer/typemaps.cpp"
  COMMAND cd "${CMAKE_CURRENT_SOURCE_DIR}" && doxygen doxygen.conf
  COMMAND ruby "${CMAKE_CURRENT_SOURCE_DIR}/../tools/typemapgenerator/generate.rb" --namespace LibGeoDecomp --header-fix "^.+/src:libgeodecomp" "${CMAKE_CURRENT_SOURCE_DIR}/../doc/xml" "${CMAKE_CURRENT_SOURCE_DIR}/mpilayer"
  DEPENDS "${CMAKE_BINARY_DIR}/cache"
)

list(REMOVE_ITEM ALL_HEADERS "mpilayer/typemaps.h")
set(CANDIDATES)
foreach(i ${ALL_HEADERS})
  set(CANDIDATES "${i}:${CANDIDATES}")
endforeach(i)

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/cache"
  COMMAND cp -a "${CMAKE_BINARY_DIR}/last_cache" "${CMAKE_BINARY_DIR}/cache" 2>/dev/null || touch "${CMAKE_BINARY_DIR}/cache"
  COMMAND ruby "${CMAKE_CURRENT_SOURCE_DIR}/../tools/typemapgenerator/generate.rb" --cache "${CMAKE_BINARY_DIR}/cache" "${CMAKE_CURRENT_SOURCE_DIR}" "${CANDIDATES}"
  COMMAND cp -a "${CMAKE_BINARY_DIR}/cache" "${CMAKE_BINARY_DIR}/last_cache"
  DEPENDS ${ALL_HEADERS}
)
