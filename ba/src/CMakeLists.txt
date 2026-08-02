# Collect all source files
file(GLOB_RECURSE BA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Create the library
add_library(ba ${BA_SOURCES})

# Include directories
target_include_directories(ba
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Path to the repo's ba_py directory, used to locate visualization scripts
# invoked at runtime (see result.cpp, loc_problem.cpp) without hard-coding a
# machine-specific path.
get_filename_component(DR_BA_REPO_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
target_compile_definitions(ba PRIVATE
    DR_BA_PY_DIR="${DR_BA_REPO_ROOT}/ba_py"
    DR_BA_APP_DIR="${CMAKE_SOURCE_DIR}/app"
)
