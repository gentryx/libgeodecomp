generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_INTRINSICS)
  add_executable(vectorization main.cpp)
  target_link_libraries(vectorization ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
