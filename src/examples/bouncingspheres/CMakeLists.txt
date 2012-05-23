generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(bouncingspheres ${SOURCES})
target_link_libraries(bouncingspheres ${LOCAL_LIBGEODECOMP_LINK_LIB})
