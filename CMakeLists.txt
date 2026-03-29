cmake_minimum_required(VERSION 3.20)
project(hatching LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ===========================================================================
# Dependencies via FetchContent
# ===========================================================================
include(FetchContent)

# --- Eigen (header-only linear algebra) ---
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)

# --- GLFW (windowing) ---
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
  GIT_SHALLOW    TRUE
)

# --- glad (pre-generated OpenGL 4.1 Core loader, vendored) ---
# Generated via: python -m glad --out-path third_party/glad --api gl:core=4.1 --reproducible c

# --- Dear ImGui (UI) ---
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.91.8
  GIT_SHALLOW    TRUE
)

# --- tinyobjloader (OBJ file loading) ---
FetchContent_Declare(tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG        v2.0.0rc13
  GIT_SHALLOW    TRUE
)

# --- Google Test (testing, test-only) ---
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(eigen glfw tinyobjloader googletest imgui)

# --- glad: build from vendored pre-generated sources ---
add_library(glad_gl_core_41 STATIC
  ${CMAKE_SOURCE_DIR}/third_party/glad/src/gl.c
)
target_include_directories(glad_gl_core_41 PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/glad/include
)

# --- Dear ImGui: build as static library (no native CMakeLists.txt) ---
find_package(OpenGL REQUIRED)

set(IMGUI_DIR ${imgui_SOURCE_DIR})
add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_demo.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
  ${IMGUI_DIR}
  ${IMGUI_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw glad_gl_core_41 OpenGL::GL)

# ===========================================================================
# Core library (algorithms, no OpenGL dependency)
# ===========================================================================
add_library(hatching_core STATIC
  src/triangle_mesh.cpp
  src/geometry.cpp
  src/direction_field.cpp
  src/stripe_pattern.cpp
)
target_include_directories(hatching_core PUBLIC src)
target_link_libraries(hatching_core PUBLIC Eigen3::Eigen tinyobjloader)

# ===========================================================================
# Main application
# ===========================================================================
add_executable(hatching
  src/main.cpp
  src/renderer.cpp
  src/camera.cpp
)
target_link_libraries(hatching PRIVATE
  hatching_core
  glfw
  glad_gl_core_41
  imgui
  OpenGL::GL
)

# Copy shaders and assets next to the binary
add_custom_command(TARGET hatching POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:hatching>/shaders
)

# ===========================================================================
# Tests
# ===========================================================================
enable_testing()
add_executable(hatching_tests
  tests/test_mesh.cpp
  tests/test_geometry.cpp
  tests/test_direction_field.cpp
  tests/test_stripe_pattern.cpp
)
target_link_libraries(hatching_tests PRIVATE hatching_core GTest::gtest_main)

# Make test data paths available
target_compile_definitions(hatching_tests PRIVATE
  TEST_DATA_DIR="${CMAKE_SOURCE_DIR}"
)

include(GoogleTest)
gtest_discover_tests(hatching_tests)
