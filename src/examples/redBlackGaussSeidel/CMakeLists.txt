generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(gausSeidel ${SOURCES})
target_link_libraries(gausSeidel ${LOCAL_LIBGEODECOMP_LINK_LIB})
