#ifndef BOX_H
#define BOX_H

#include "collider.h"
#include <Eigen/Dense>
#include "mass.h"
#include <iostream>

class Box : public Collider {
public:
    Eigen::Vector3f center;  // Box center point
    Eigen::Vector3f dimensions;  // Dimensions for width, height, and depth

    Box(const Eigen::Vector3f& center, const Eigen::Vector3f& dimensions) 
        : center(center), dimensions(dimensions) {}

    bool checkCollision(Mass& mass) const override {
        Eigen::Vector3f position = mass.position;

        // TODO: Task 5 - Implement collision detection for a box

        // Calculate the half extents of the box
        Eigen::Vector3f half_extents = dimensions / 2.0;

        // Calculate min and max corners of the box
        Eigen::Vector3f min_corner = center - half_extents;
        Eigen::Vector3f max_corner = center + half_extents;

        // Check if the position of the mass is within the box
        return (position.x() >= min_corner.x() && position.x() <= max_corner.x() &&
                position.y() >= min_corner.y() && position.y() <= max_corner.y() &&
                position.z() >= min_corner.z() && position.z() <= max_corner.z());
        // return false;
    }

    void resolveCollision(Mass& mass) const override {

        // TODO: Task 5 - Resolve collision with a box
        // Reset the position to the last position before the collision
        mass.position = mass.last_position;

    }
};

#endif // BOX_H
