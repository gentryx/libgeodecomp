add_subdirectory(test/unit)

include(auto.cmake)

install(FILES ${HEADERS} DESTINATION "${INCLUDE_INSTALL_DIR}/testlib")

