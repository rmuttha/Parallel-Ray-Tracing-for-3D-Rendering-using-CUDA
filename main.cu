// ******************Parallel Ray Tracing for 3D Rendering using CUDA****************************
// Author: Rutuja Muttha
// Project Details: Description: This CUDA program implements a parallel ray tracing algorithm for 3D rendering.
//                  The program initializes a scene with basic geometric shapes (e.g., spheres) and 
//                  uses ray tracing to calculate pixel colors based on light interactions with 
//                  the objects. The rendering is performed in parallel on the GPU to enhance 
//                  performance and achieve realistic images. 
// File Details: The main CUDA file where the ray tracing algorithm is implemented.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include "scene.h"
#include "camera.h"
#include "ray.h"

#define WIDTH 800
#define HEIGHT 600
#define SAMPLES 10

__global__ void renderScene(float3* pixels, Camera camera, Scene scene) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int index = y * WIDTH + x;
    float3 color = make_float3(0, 0, 0);

    for (int i = 0; i < SAMPLES; ++i) {
        Ray ray = camera.generateRay(x, y);
        color += scene.trace(ray, 0);
    }

    color /= SAMPLES;
    pixels[index] = color;
}

int main() {
    // Allocate memory for the rendered image
    float3* d_pixels;
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * sizeof(float3));

    // Initialize camera and scene
    Camera camera(make_float3(0, 0, 1), make_float3(0, 0, -1), make_float3(0, 1, 0), 90.0f, float(WIDTH) / HEIGHT);
    Scene scene;
    scene.addObject(new Sphere(make_float3(0, 0, -5), 1.0f, make_float3(1, 0, 0))); // Add a red sphere

    // Launch the render kernel
    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    renderScene<<<grid, block>>>(d_pixels, camera, scene);
    cudaDeviceSynchronize();

    // Copy results back to host and save the image
    float3* h_pixels = new float3[WIDTH * HEIGHT];
    cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);

    // Save image code here (PPM format for simplicity)

    // Clean up
    cudaFree(d_pixels);
    delete[] h_pixels;

    return 0;
}
