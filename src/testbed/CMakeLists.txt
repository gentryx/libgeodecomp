generate_sourcelists("./")

add_subdirectory(interfaces)

set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(testbed ${SOURCES})
target_link_libraries(testbed ${LOCAL_LIBGEODECOMP_LINK_LIB})
