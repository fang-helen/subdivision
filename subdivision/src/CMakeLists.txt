SET(pwd ${CMAKE_CURRENT_LIST_DIR})

SET(src "")
AUX_SOURCE_DIRECTORY(${pwd} src)
add_executable(menger ${src})
message(STATUS "menger added")

target_link_libraries(menger ${stdgl_libraries})
