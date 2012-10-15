generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_INTRINSICS)
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  add_executable(performancetests ${SOURCES})
  target_link_libraries(performancetests ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
