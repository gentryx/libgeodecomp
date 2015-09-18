lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI AND WITH_INTRINSICS)
  add_executable(libgeodecomp_examples_jacobi3dupdateline ${SOURCES})
  set_target_properties(libgeodecomp_examples_jacobi3dupdateline PROPERTIES OUTPUT_NAME jacobi3dupdateline)
  target_link_libraries(libgeodecomp_examples_jacobi3dupdateline ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
