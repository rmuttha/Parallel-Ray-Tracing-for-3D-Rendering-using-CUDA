// ******************Parallel Ray Tracing for 3D Rendering using CUDA****************************
// Author: Rutuja Muttha
// Project Details: Description: This CUDA program implements a parallel ray tracing algorithm for 3D rendering.
//                  The program initializes a scene with basic geometric shapes (e.g., spheres) and 
//                  uses ray tracing to calculate pixel colors based on light interactions with 
//                  the objects. The rendering is performed in parallel on the GPU to enhance 
//                  performance and achieve realistic images. 
// File Details: 	Header file containing definitions for the scene setup, including objects and materials.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef SCENE_H
#define SCENE_H

#include "ray.h"
#include "material.h"

class Scene {
public:
    __device__ float3 trace(const Ray& ray, int depth);
    void addObject(Object* object);

private:
    Object* objects[10]; // Simplified for demo; dynamic allocation preferred
    int objectCount = 0;
};

#endif
