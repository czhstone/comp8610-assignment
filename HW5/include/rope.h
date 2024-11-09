#ifndef ROPE_H
#define ROPE_H

#include <vector>
#include "mass.h"
#include "spring.h"
#include <Eigen/Dense>

enum class IntegrationMethod {
    EXPLICIT,
    IMPLICIT
};

class Rope {
public:
    Rope(Eigen::Vector3f start, Eigen::Vector3f end, int num_nodes, float node_mass, float k, float damping_factor, std::vector<int> fixed_nodes);

    void simulateVerlet(float delta_t, Eigen::Vector3f gravity);
    void simulateEuler(float delta_t, Eigen::Vector3f gravity, IntegrationMethod method);

    float damping_factor;
    std::vector<Mass*> masses;
    std::vector<Spring*> springs;

private:
    void initializeMasses(Eigen::Vector3f start, Eigen::Vector3f end, int num_nodes, float node_mass, std::vector<int> fixed_nodes);
    void linkSprings(float k);
};

#endif /* ROPE_H */
