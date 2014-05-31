generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(redBlackGaussSeidel ${SOURCES})
target_link_libraries(redBlackGaussSeidel ${LOCAL_LIBGEODECOMP_LINK_LIB})
