#ifndef MASS_H
#define MASS_H

#include <Eigen/Dense>

struct Mass{
    Mass(Eigen::Vector3f position, float mass, bool is_fixed = false)
        : start_position(position), position(position), last_position(position),
            velocity(Eigen::Vector3f(0, 0, 0)), force(Eigen::Vector3f(0, 0, 0)),
            mass(mass), is_fixed(is_fixed) {}

    Eigen::Vector3f start_position;
    Eigen::Vector3f position;

    // Verlet
    Eigen::Vector3f last_position;

    // Euler
    Eigen::Vector3f velocity;
    Eigen::Vector3f force;

    float mass;
    bool is_fixed;
};

#endif /* MASS_H */