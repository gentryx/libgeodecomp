generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

set(CMAKE_CXX_FLAGS "-O3 -march=native -g -DMY_SIZE=7512")

add_executable(interfaces ${SOURCES})
target_link_libraries(interfaces ${LOCAL_LIBGEODECOMP_LINK_LIB})
