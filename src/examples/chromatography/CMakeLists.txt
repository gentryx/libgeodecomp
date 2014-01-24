generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_SILO)
  add_executable(chromatography ${SOURCES})
  target_link_libraries(chromatography ${LOCAL_LIBGEODECOMP_LINK_LIB} hdf5 siloh5 z)
endif(WITH_SILO)
