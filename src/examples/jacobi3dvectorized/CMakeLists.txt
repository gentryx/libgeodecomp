generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI)
  add_executable(libgeodecomp_examples_jacobi3dvectorized ${SOURCES})
  set_target_properties(libgeodecomp_examples_jacobi3dvectorized PROPERTIES OUTPUT_NAME jacobi3dvectorized)
  target_link_libraries(libgeodecomp_examples_jacobi3dvectorized ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
