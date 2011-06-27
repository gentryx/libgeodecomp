set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(ants ${SOURCES})
target_link_libraries(ants ${LOCAL_LIBGEODECOMP_LINK_LIB})
