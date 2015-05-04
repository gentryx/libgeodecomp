generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI)
  add_executable(libgeodecomp_examples_wing ${SOURCES})
  set_target_properties(libgeodecomp_examples_wing PROPERTIES OUTPUT_NAME wing)
  target_link_libraries(libgeodecomp_examples_wing ${LOCAL_LIBGEODECOMP_LINK_LIB} geodecomp)
endif(WITH_MPI)
