// ******************Parallel Ray Tracing for 3D Rendering using CUDA****************************
// Author: Rutuja Muttha
// Project Details: Description: This CUDA program implements a parallel ray tracing algorithm for 3D rendering.
//                  The program initializes a scene with basic geometric shapes (e.g., spheres) and 
//                  uses ray tracing to calculate pixel colors based on light interactions with 
//                  the objects. The rendering is performed in parallel on the GPU to enhance 
//                  performance and achieve realistic images. 
// File Details: CUDA file implementing the ray operations.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ray.h"

// Since the Ray class is simple, it doesn't require extensive implementation beyond its basic constructor,
// which is already defined in the header file. The operations are usually direct manipulations of the ray's
// origin and direction vectors, which are utilized in the intersection calculations within the Scene.

__device__ Ray::Ray(float3 origin, float3 direction)
    : origin(origin), direction(direction) {
    // Normalizing the direction of the ray
    float magnitude = sqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    this->direction = make_float3(direction.x / magnitude, direction.y / magnitude, direction.z / magnitude);
}

// Additional helper functions for the Ray class could be added here, such as reflection or refraction functions,
// which are useful for implementing complex lighting models, but are not essential for the basic ray tracing operation.

