cmake_minimum_required(VERSION 3.0.0)
project(paper_manipulator VERSION 0.1.0)

set(BULLET_PHYSICS_SOURCE_DIR, bullet3)

INCLUDE_DIRECTORIES(
${BULLET_PHYSICS_SOURCE_DIR}/src
)

LINK_LIBRARIES(
 BulletDynamics BulletCollision LinearMath
)

include(CTest)
enable_testing()

add_executable(paper_manipulator src/paper_dynamics.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
