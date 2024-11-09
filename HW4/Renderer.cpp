//
// Created by goksu on 2/25/20.
//

#include <fstream>
#include <sstream>
#include "Scene.hpp"
#include "Renderer.hpp"
#include "Material.hpp"
#ifdef _OPENMP
    #include <omp.h>
#endif


inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

const float EPSILON = 1e-2;

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height);

    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    Vector3f eye_pos(278, 273, -800);

    std::cout << "SPP: " << scene.spp << "\n";

    float progress = 0.0f;

#pragma omp parallel for num_threads(8) // use multi-threading for speedup if openmp is available
    for (uint32_t j = 0; j < scene.height; ++j) {
        for (uint32_t i = 0; i < scene.width; ++i) {

            int m = i + j * scene.width;
            if(scene.spp==1){
                // TODO: task 1.2 pixel projection

                // Convert pixel indices `i` and `j` to floating-point numbers for precise calculations
                float pixel_i = static_cast<float>(i) + 0.5f;
                float pixel_j = static_cast<float>(j) + 0.5f;

                // Calculate normalized coordinates within the range [-1, 1]
                float x = (2.0f * pixel_i / static_cast<float>(scene.width) - 1.0f) * imageAspectRatio * scale;
                float y = (1.0f - 2.0f * pixel_j / static_cast<float>(scene.height)) * scale;

                // Construct the ray direction from the eye position to the point on the image plane
                Vector3f dir = normalize(Vector3f(-x, y, 1));
                framebuffer[m] = scene.castRay(Ray(eye_pos, dir), 0);

            }else {
                // TODO: task 4 multi-sampling
                Vector3f accumulatedColor = Vector3f(0.0f, 0.0f, 0.0f);  // Initialize the color accumulator to hold the final pixel color

                float samplesPerPixel = sqrt(scene.spp);  // Calculate the number of samples per axis based on the square root of samples per pixel

                // Loop through the rows of sub-pixels (samples)
                for (int sampleRow = 0; sampleRow < samplesPerPixel; sampleRow++) {
                    // Loop through the columns of sub-pixels (samples)
                    for (int sampleCol = 0; sampleCol < samplesPerPixel; sampleCol++) {
                        // Compute the horizontal and vertical offsets within the current pixel
                        float horizontalOffset = (sampleCol + 0.5f) / samplesPerPixel;  // X-axis offset
                        float verticalOffset = (sampleRow + 0.5f) / samplesPerPixel;    // Y-axis offset

                        // Normalize the pixel's offset coordinates to the range [-1, 1] while applying scaling
                        float normalizedX = ((static_cast<float>(i) + horizontalOffset) / static_cast<float>(scene.width) * 2.0f - 1.0f) * imageAspectRatio * scale;
                        float normalizedY = (1.0f - (static_cast<float>(j) + verticalOffset) / static_cast<float>(scene.height) * 2.0f) * scale;

                        // Calculate the ray's direction vector by normalizing towards the view plane
                        Vector3f rayDirection = normalize(Vector3f(-normalizedX, normalizedY, 1));

                        // Trace a ray from the camera's position using the calculated direction and accumulate the color
                        accumulatedColor += scene.castRay(Ray(eye_pos, rayDirection), 0);
                    }
                }
                framebuffer[m] = accumulatedColor / static_cast<float>(scene.spp);
            }
        }
        progress += 1.0f / (float)scene.height;
        UpdateProgress(progress);
    }
    UpdateProgress(1.f);

    // save framebuffer to file
    std::stringstream ss;
    ss << "binary_task" << TASK_N<<".ppm";
    std::string str = ss.str();
    const char* file_name = str.c_str();
    FILE* fp = fopen(file_name, "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);
}
