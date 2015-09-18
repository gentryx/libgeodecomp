lgd_generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI)
  add_executable(libgeodecomp_testbed_parallelperformancetests ${SOURCES})
  set_target_properties(libgeodecomp_testbed_parallelperformancetests PROPERTIES OUTPUT_NAME parallelperformancetests)
  target_link_libraries(libgeodecomp_testbed_parallelperformancetests ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()