lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_CPP14)
  add_executable(libgeodecomp_examples_spmvm ${SOURCES})
  set_target_properties(libgeodecomp_examples_spmvm PROPERTIES OUTPUT_NAME spmv)
  target_link_libraries(libgeodecomp_examples_spmvm ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
