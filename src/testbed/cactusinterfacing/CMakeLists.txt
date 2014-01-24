generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI)
  add_executable(cactusinterfacing main.cpp)
  target_link_libraries(cactusinterfacing ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()