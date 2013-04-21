generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_MPI)
  add_executable(latticeboltzmann ${SOURCES})
  target_link_libraries(latticeboltzmann ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif(FEATURE_MPI)
