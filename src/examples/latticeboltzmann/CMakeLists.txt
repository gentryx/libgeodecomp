generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if (LIBGEODECOMP_FEATURE_MPI)
  add_executable(lbm ${SOURCES})
  target_link_libraries(lbm ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif (LIBGEODECOMP_FEATURE_MPI)
