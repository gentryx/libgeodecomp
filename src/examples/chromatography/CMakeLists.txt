generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_SILO)
  add_executable(chromatography ${SOURCES})
  target_link_libraries(chromatography ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif(WITH_SILO)
