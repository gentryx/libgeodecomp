generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_SILO)
  add_executable(voronoi ${SOURCES})
  target_link_libraries(voronoi ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
