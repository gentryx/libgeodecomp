generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_MPI)
  add_executable(wing ${SOURCES})
  target_link_libraries(wing ${LOCAL_LIBGEODECOMP_LINK_LIB} geodecomp)
endif(FEATURE_MPI)
