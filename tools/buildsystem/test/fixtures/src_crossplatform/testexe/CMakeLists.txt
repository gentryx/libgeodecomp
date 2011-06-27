include(auto.cmake)

add_executable(testexe ${SOURCES})
target_link_libraries (testexe superlib)
install(TARGETS testexe DESTINATION bin)
