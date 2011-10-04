set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(latticegas ${SOURCES})
target_link_libraries(latticegas ${LOCAL_LIBGEODECOMP_LINK_LIB})
