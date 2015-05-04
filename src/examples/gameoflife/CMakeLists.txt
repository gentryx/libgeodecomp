generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI)
  add_executable(libgeodecomp_examples_gameoflife ${SOURCES})
  set_target_properties(libgeodecomp_examples_gameoflife PROPERTIES OUTPUT_NAME gameoflife)
  target_link_libraries(libgeodecomp_examples_gameoflife ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif(WITH_MPI)
