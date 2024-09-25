////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ******************Parallel Ray Tracing for 3D Rendering using CUDA****************************
// Author: Rutuja Muttha
// Project Details: Description: This CUDA program implements a parallel ray tracing algorithm for 3D rendering.
//                  The program initializes a scene with basic geometric shapes (e.g., spheres) and 
//                  uses ray tracing to calculate pixel colors based on light interactions with 
//                  the objects. The rendering is performed in parallel on the GPU to enhance 
//                  performance and achieve realistic images. 
// File Details: Header file defining the Camera class. 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

class Camera {
public:
    __device__ Camera(float3 position, float3 lookAt, float3 up, float fov, float aspect);
    __device__ Ray generateRay(int x, int y);

private:
    float3 position;
    float3 lowerLeftCorner;
    float3 horizontal;
    float3 vertical;
};

#endif
