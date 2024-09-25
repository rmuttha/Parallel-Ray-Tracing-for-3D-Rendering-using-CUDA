////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ******************Parallel Ray Tracing for 3D Rendering using CUDA****************************
// Author: Rutuja Muttha
// Project Details: Description: This CUDA program implements a parallel ray tracing algorithm for 3D rendering.
//                  The program initializes a scene with basic geometric shapes (e.g., spheres) and 
//                  uses ray tracing to calculate pixel colors based on light interactions with 
//                  the objects. The rendering is performed in parallel on the GPU to enhance 
//                  performance and achieve realistic images. 
// File Details: Header file defining materials for objects 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef MATERIAL_H
#define MATERIAL_H

#include <cuda_runtime.h>

struct Material {
    float3 color;        // Base color of the material
    float reflection;    // Reflectivity factor

    __device__ Material(float3 color, float reflection) : color(color), reflection(reflection) {}
};

#endif
