#include "include/rasterizer.h"
#include <cmath>
#include "include/transforms.h"
#include "include/rope.h"
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
using namespace cv;


int main(int argc, char** argv) {

    // Configuration for the simulation
    struct {
        // Cloth properties
        int num_nodes_x = 16;
        int num_nodes_z = 16;

        float node_mass = 1;
        float k = 100;
        float damping_factor = 0.001;

        // Environment variables
        Eigen::Vector3f gravity = {0, -0.2, 0}; // Do not change
        int steps_per_fps = 64; // Number of simulation steps per frame
        int fps = 60;

    } config;

    std::cout << "Press ESC to exit the simulation" << std::endl;

    rasterizer r(800, 800);

    Eigen::Vector3f center = {0, 5, 0};
    Eigen::Vector2f size = {10, 10};

    Cloth cloth(center,
                size, 
                config.num_nodes_x,
                config.num_nodes_z,
                config.node_mass,
                config.k,
                config.damping_factor);

    cloth.fixMass(0);
    cloth.fixMass(config.num_nodes_x - 1);

    const char* windowName = "Image Window";
    namedWindow(windowName);

    cv::Point2f delta = cv::Point2f(1, 0);

    Eigen::Vector3f eye_pos = {0, 0, 20};

    float angle = 0;
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));
    r.set_model(get_model_matrix(angle, {0, 1, 0}, {0, 0, 0}));


    while (true) {
        auto start = std::chrono::high_resolution_clock::now();


        for (int i = 0; i < config.steps_per_fps; ++i) {
            cloth.simulateVerlet(1.0/config.steps_per_fps, config.gravity);
        }   

        r.clear();

        r.drawCloth(&cloth, {255, 255, 255});

        r.flip_vertically();
        imshow(windowName, r.to_mat());

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        int remaining_time = 1000000 / config.fps - duration.count();
        if (remaining_time > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(remaining_time));
        }

        if (waitKey(1) == 27) {
            break;
        }
    }

    destroyWindow(windowName);
    return 0;
}
