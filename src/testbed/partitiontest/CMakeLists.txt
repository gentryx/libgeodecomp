generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_INTRINSICS)
  add_executable(partitiontest main.cpp)
  target_link_libraries(partitiontest ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
