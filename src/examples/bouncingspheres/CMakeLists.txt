set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(granulargas ${SOURCES})
target_link_libraries(granulargas ${LOCAL_LIBGEODECOMP_LINK_LIB})
