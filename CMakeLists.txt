cmake_minimum_required(VERSION 3.31)
project(ProyectoIntercicloDACP)

# Estándar de C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ruta personalizada a tu OpenCV compilado desde fuente
set(OpenCV_DIR "/home/diego/Aplicaciones/Librerias/opencv/opencvi/lib/cmake/opencv5")

# Buscar OpenCV e ITK
find_package(OpenCV REQUIRED)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Incluir directorios
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        /home/diego/Aplicaciones/Librerias/opencv/opencvi/include/opencv5
)

# Archivos fuente
add_executable(ProyectoIntercicloDACP main.cpp)

# Enlazar librerías OpenCV e ITK
target_link_libraries(ProyectoIntercicloDACP
        ${OpenCV_LIBS}
        ${ITK_LIBRARIES}
)
