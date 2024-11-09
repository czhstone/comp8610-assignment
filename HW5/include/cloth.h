#ifndef CLOTH_H
#define CLOTH_H

#include <vector>
#include "mass.h"
#include "spring.h"
#include "collider.h"

#include <Eigen/Dense>

class Cloth {
public:

    int num_nodes_x, num_nodes_z;
    std::vector<Mass*> masses;

    Cloth(const Eigen::Vector3f& center, const Eigen::Vector2f& size, int num_nodes_x, int num_nodes_z, float node_mass, float k, float damping_factor);

    void simulateVerlet(float delta_t, Eigen::Vector3f gravity);
    void fixMass(int i);
    
    void addCollider(Collider* collider);

private:
    Eigen::Vector3f center;
    Eigen::Vector2f size;
    float node_mass;
    float spring_constant;
    float damping_factor;
    std::vector<Spring*> springs;
    std::vector<Collider*> colliders;

    void initializeMasses();
    void linkSprings();
    
};

#endif /* CLOTH_H */
