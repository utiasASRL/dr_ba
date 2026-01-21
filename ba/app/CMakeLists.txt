# Interface library for common dependencies
add_library(dr_common INTERFACE)

target_include_directories(dr_common
    INTERFACE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(dr_common
    INTERFACE
        ba
        yaml-cpp
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX   # handles both compile & link flags automatically
)

# Executables
add_executable(dr_ba dr_ba.cpp)
add_executable(dr_map dr_map.cpp)
add_executable(dr_loc dr_loc.cpp)

# Link all executables to the common interface
target_link_libraries(dr_ba PRIVATE dr_common)
target_link_libraries(dr_map PRIVATE dr_common)
target_link_libraries(dr_loc PRIVATE dr_common)
