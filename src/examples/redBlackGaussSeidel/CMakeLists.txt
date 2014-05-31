generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(gaussSeidel ${SOURCES})
target_link_libraries(gaussSeidel ${LOCAL_LIBGEODECOMP_LINK_LIB})
