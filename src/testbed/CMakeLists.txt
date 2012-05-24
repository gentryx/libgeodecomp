generate_sourcelists("./")

add_subdirectory(interfaces)

set(RELATIVE_PATH "")
include(auto.cmake)

# fixme: won't work on woody
set(CMAKE_CXX_FLAGS "-O3 -march=native -g -DMY_SIZE=7512")

add_executable(testbed ${SOURCES})
target_link_libraries(testbed ${LOCAL_LIBGEODECOMP_LINK_LIB})
