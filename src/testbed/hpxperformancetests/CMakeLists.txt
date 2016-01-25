lgd_generate_sourcelists("./")
set(RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/")
include(auto.cmake)

if(WITH_HPX)
  add_executable(hpxperformancetests ${SOURCES})
  target_link_libraries(hpxperformancetests ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
