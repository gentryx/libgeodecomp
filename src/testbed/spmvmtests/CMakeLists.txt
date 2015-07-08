lgd_generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_CPP14 AND WITH_INTRINSICS)
  add_executable(libgeodecomp_testbed_spmvmtests main.cpp mmio.cpp)
  set_target_properties(libgeodecomp_testbed_spmvmtests PROPERTIES OUTPUT_NAME spmvmtests)
  target_link_libraries(libgeodecomp_testbed_spmvmtests ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
