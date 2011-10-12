set(RELATIVE_PATH "")
include(auto.cmake)

if (LIBGEODECOMP_FEATURE_MPI)
  add_executable(wing ${SOURCES})
  target_link_libraries(wing ${LOCAL_LIBGEODECOMP_LINK_LIB} geodecomp)
endif (LIBGEODECOMP_FEATURE_MPI)
