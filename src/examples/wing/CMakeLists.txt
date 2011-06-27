set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(wing ${SOURCES})
target_link_libraries(wing ${LOCAL_LIBGEODECOMP_LINK_LIB} geodecomp)
