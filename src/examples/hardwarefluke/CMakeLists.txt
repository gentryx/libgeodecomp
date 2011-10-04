set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(hardwarefluke ${SOURCES})
target_link_libraries(hardwarefluke ${LOCAL_LIBGEODECOMP_LINK_LIB})
