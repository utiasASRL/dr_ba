add_executable(dr_ba dr_ba.cpp)

target_link_libraries(dr_ba
    PRIVATE
        ba
        yaml-cpp
        ${OpenCV_LIBS}
)

target_include_directories(dr_ba
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
