lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI)
  add_executable(libgeodecomp_examples_latticeboltzmann ${SOURCES})
  set_target_properties(libgeodecomp_examples_latticeboltzmann PROPERTIES OUTPUT_NAME latticeboltzmann)
  target_link_libraries(libgeodecomp_examples_latticeboltzmann ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif(WITH_MPI)
