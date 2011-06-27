set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(gameoflife ${SOURCES})
target_link_libraries(gameoflife ${LOCAL_LIBGEODECOMP_LINK_LIB})
