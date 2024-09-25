// ******************Parallel Ray Tracing for 3D Rendering using CUDA****************************
// Author: Rutuja Muttha
// Project Details: Description: This CUDA program implements a parallel ray tracing algorithm for 3D rendering.
//                  The program initializes a scene with basic geometric shapes (e.g., spheres) and 
//                  uses ray tracing to calculate pixel colors based on light interactions with 
//                  the objects. The rendering is performed in parallel on the GPU to enhance 
//                  performance and achieve realistic images. 
// File Details: CUDA file that initializes the scene with objects like spheres and planes. 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#include "scene.h"

__device__ float3 Scene::trace(const Ray& ray, int depth) {
    float3 hitColor = make_float3(0, 0, 0); // Default background color
    float tMin = 1e20;
    Object* hitObject = nullptr;

    for (int i = 0; i < objectCount; ++i) {
        float t = objects[i]->intersect(ray);
        if (t > 0 && t < tMin) {
            tMin = t;
            hitObject = objects[i];
        }
    }

    if (hitObject) {
        hitColor = hitObject->material.color; // Simplified shading model
    }

    return hitColor;
}

void Scene::addObject(Object* object) {
    objects[objectCount++] = object;
}
