lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_CPP14)
  add_executable(libgeodecomp_examples_spmvmvectorized ${SOURCES})
  set_target_properties(libgeodecomp_examples_spmvmvectorized PROPERTIES OUTPUT_NAME spmvmvectorized)
  target_link_libraries(libgeodecomp_examples_spmvmvectorized ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
