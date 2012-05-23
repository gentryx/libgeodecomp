generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if (LIBGEODECOMP_FEATURE_MPI)
  add_executable(gameoflife ${SOURCES})
  target_link_libraries(gameoflife ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif (LIBGEODECOMP_FEATURE_MPI)
