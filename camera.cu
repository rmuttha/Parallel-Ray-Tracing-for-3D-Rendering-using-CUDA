// ******************Parallel Ray Tracing for 3D Rendering using CUDA****************************
// Author: Rutuja Muttha
// Project Details: Description: This CUDA program implements a parallel ray tracing algorithm for 3D rendering.
//                  The program initializes a scene with basic geometric shapes (e.g., spheres) and 
//                  uses ray tracing to calculate pixel colors based on light interactions with 
//                  the objects. The rendering is performed in parallel on the GPU to enhance 
//                  performance and achieve realistic images. 
// File Details: CUDA file implementing camera operations, including ray generation.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "camera.h"

__device__ Camera::Camera(float3 position, float3 lookAt, float3 up, float fov, float aspect) {
    position = position;
    float theta = fov * 3.14159265359 / 180.0;
    float halfHeight = tan(theta / 2);
    float halfWidth = aspect * halfHeight;
    lowerLeftCorner = position - make_float3(halfWidth, halfHeight, 1.0);
    horizontal = make_float3(2 * halfWidth, 0, 0);
    vertical = make_float3(0, 2 * halfHeight, 0);
}

__device__ Ray Camera::generateRay(int x, int y) {
    float u = float(x) / float(WIDTH);
    float v = float(y) / float(HEIGHT);
    float3 direction = lowerLeftCorner + u * horizontal + v * vertical - position;
    return Ray(position, direction);
}
