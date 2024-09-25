// ******************Parallel Ray Tracing for 3D Rendering using CUDA****************************
// Author: Rutuja Muttha
// Project Details: Description: This CUDA program implements a parallel ray tracing algorithm for 3D rendering.
//                  The program initializes a scene with basic geometric shapes (e.g., spheres) and 
//                  uses ray tracing to calculate pixel colors based on light interactions with 
//                  the objects. The rendering is performed in parallel on the GPU to enhance 
//                  performance and achieve realistic images. 
// File Details: Header file defining the Ray class and related functions.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#ifndef RAY_H
#define RAY_H

#include <cuda_runtime.h>

class Ray {
public:
    __device__ Ray(float3 origin, float3 direction) : origin(origin), direction(direction) {}
    float3 origin;
    float3 direction;
};

#endif
