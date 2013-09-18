generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_MPI)
  add_executable(parallelperformancetests ${SOURCES})
  target_link_libraries(parallelperformancetests ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()