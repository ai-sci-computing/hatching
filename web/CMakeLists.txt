cmake_minimum_required(VERSION 3.20)
project(hatching_web LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===========================================================================
# Dependencies
# ===========================================================================
include(FetchContent)

# --- Eigen (header-only) ---
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

# ===========================================================================
# hatching_core — reuse sources from parent project (read-only)
# ===========================================================================
set(CORE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_library(hatching_core STATIC
  ${CORE_SRC_DIR}/triangle_mesh.cpp
  ${CORE_SRC_DIR}/geometry.cpp
  ${CORE_SRC_DIR}/direction_field.cpp
  ${CORE_SRC_DIR}/stripe_pattern.cpp
)
target_include_directories(hatching_core PUBLIC ${CORE_SRC_DIR})
target_link_libraries(hatching_core PUBLIC Eigen3::Eigen)

# The desktop build links tinyobjloader for file-based OBJ loading.
# The web bridge parses OBJ from strings directly, so we provide a stub
# tiny_obj_loader.h that satisfies the #include in triangle_mesh.cpp.
# The stub include dir must come BEFORE any system paths.
target_include_directories(hatching_core BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ===========================================================================
# WASM module
# ===========================================================================
add_executable(hatching_web src/bridge.cpp)
target_link_libraries(hatching_web PRIVATE hatching_core)

# Emscripten link flags
target_link_options(hatching_web PRIVATE
  "SHELL:-s WASM=1"
  "SHELL:-s ALLOW_MEMORY_GROWTH=1"
  "SHELL:-s MODULARIZE=1"
  "SHELL:-s EXPORT_NAME='HatchingModule'"
  "SHELL:-s EXPORT_ES6=1"
  "SHELL:-s ENVIRONMENT='web'"
  "SHELL:--bind"
  "SHELL:-s INITIAL_MEMORY=67108864"
  "SHELL:-s STACK_SIZE=5242880"
)

set_target_properties(hatching_web PROPERTIES
  OUTPUT_NAME "hatching"
  SUFFIX ".js"
)

# Copy web assets next to the output — always runs so JS/HTML edits are picked up.
add_custom_target(copy_web_assets ALL
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/index.html
    $<TARGET_FILE_DIR:hatching_web>/index.html
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/js
    $<TARGET_FILE_DIR:hatching_web>/js
  DEPENDS hatching_web
  COMMENT "Copying web assets"
)
