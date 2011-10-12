set(RELATIVE_PATH "")
include(auto.cmake)

if (LIBGEODECOMP_FEATURE_SILO)
  add_executable(chromo ${SOURCES})
  target_link_libraries(chromo ${LOCAL_LIBGEODECOMP_LINK_LIB} hdf5 siloh5 z)
endif (LIBGEODECOMP_FEATURE_SILO)
