lgd_generate_sourcelists("./")
set(RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/")
include(auto.cmake)

if(WITH_HPX)
  add_executable(libgeodecomp_testbed_hpxperformancetests ${SOURCES})
  set_target_properties(libgeodecomp_testbed_hpxperformancetests PROPERTIES OUTPUT_NAME hpxperformancetests)
  target_link_libraries(libgeodecomp_testbed_hpxperformancetests ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
