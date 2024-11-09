#include "include/cloth.h"
#include <iostream>

Cloth::Cloth(const Eigen::Vector3f& center, const Eigen::Vector2f& size, int num_nodes_x, int num_nodes_z, float node_mass, float k, float damping_factor)
    : center(center), size(size), num_nodes_x(num_nodes_x), num_nodes_z(num_nodes_z), node_mass(node_mass), spring_constant(k), damping_factor(damping_factor) {
    initializeMasses();
    linkSprings();
}

void Cloth::initializeMasses() {
    // Calculate the top left and bottom right corners of the cloth
    Eigen::Vector3f top_left = Eigen::Vector3f(center.x() - size.x() / 2, center.y(), center.z() - size.y() / 2);
    Eigen::Vector3f bottom_right = Eigen::Vector3f(center.x() + size.x() / 2, center.y(), center.z() + size.y() / 2);


    // TODO: Task 4 - Initialize masses
    // Calculate the step size between nodes
    float step_x = size.x() / (num_nodes_x - 1);
    float step_z = size.y() / (num_nodes_z - 1);

    // Create masses in a grid in the x-z plane, all at the same y level (center.y)
    for (int z = 0; z < num_nodes_z; z++) {
        for (int x = 0; x < num_nodes_x; x++) {
            Eigen::Vector3f position = Eigen::Vector3f(top_left.x() + x * step_x, center.y(), top_left.z() + z * step_z);
            Mass* mass = new Mass(position, node_mass, false); // Not fixed by default
            masses.push_back(mass);
        }
    }

}



void Cloth::linkSprings() {

    // TODO: Task 4 - Link springs
    // Link structural springs
    for (int i = 0; i < num_nodes_z; ++i) {
        for (int j = 0; j < num_nodes_x; ++j) {
            int index = i * num_nodes_x + j;
            if (j < num_nodes_x - 1) {  // Right neighbor
                springs.push_back(new Spring(masses[index], masses[index + 1], spring_constant));
            }
            if (i < num_nodes_z - 1) {  // Down neighbor
                springs.push_back(new Spring(masses[index], masses[index + num_nodes_x], spring_constant));
            }
        }
    }

    // Link shear springs
    for (int i = 0; i < num_nodes_z - 1; ++i) {
        for (int j = 0; j < num_nodes_x - 1; ++j) {
            int index = i * num_nodes_x + j;
            springs.push_back(new Spring(masses[index], masses[index + num_nodes_x + 1], spring_constant)); // Down-right diagonal
            if (j > 0) {
                springs.push_back(new Spring(masses[index], masses[index + num_nodes_x - 1], spring_constant)); // Down-left diagonal
            }
        }
    }

    // Link flexion springs
    for (int i = 0; i < num_nodes_z; ++i) {
        for (int j = 0; j < num_nodes_x; ++j) {
            int index = i * num_nodes_x + j;
            if (j < num_nodes_x - 2) {  // Right neighbor two steps away
                springs.push_back(new Spring(masses[index], masses[index + 2], spring_constant));
            }
            if (i < num_nodes_z - 2) {  // Down neighbor two steps away
                springs.push_back(new Spring(masses[index], masses[index + 2 * num_nodes_x], spring_constant));
            }
        }
    }

}


void Cloth::simulateVerlet(float delta_t, Eigen::Vector3f gravity) {
    // TODO: Task 5 - Implement Verlet integration with collision handling


    // After copying your verlet integration code from Task 3, use the following code for collision handling:

    // for (auto &mass : masses) {
    //     // Your other code here...

    //     for (auto collider : colliders) {
    //         if (collider->checkCollision(*mass)) {
    //             collider->resolveCollision(*mass); 
    //         }
    //     }
    // }
    // First, update all spring forces
    for (auto &s : springs) {
        Eigen::Vector3f ab = s->m2->position - s->m1->position;
        Eigen::Vector3f force = s->k * (ab.normalized()) * (ab.norm() - s->rest_length);

        s->m1->force += force;
        s->m2->force -= force;  // Apply Newton's third law
    }

    // Update positions of all masses and handle collisions
    for (auto &m : masses) {
        if (!m->is_fixed) {
            // Add gravity force
            m->force += gravity * m->mass;

            // Calculate the acceleration from the resultant force
            Eigen::Vector3f acceleration = m->force / m->mass;

            // Store the current position to update the last position later
            Eigen::Vector3f temp_position = m->position;

            // Verlet integration formula with damping for position update
            m->position = m->position + (1 - damping_factor) * (m->position - m->last_position) + acceleration * delta_t * delta_t;

            // Update last_position to the stored current position for the next timestep
            m->last_position = temp_position;

            // Collision handling
            for (auto collider : colliders) {
                if (collider->checkCollision(*m)) {
                    collider->resolveCollision(*m); 
                }
            }
        }

        // Reset forces after calculations for the next timestep
        m->force = Eigen::Vector3f(0, 0, 0);
    }

}

void Cloth::fixMass(int i) {

    if (i < 0 || i >= masses.size()) {
        std::cerr << "Invalid mass index" << std::endl;
        return;
    }
    
    masses[i]->is_fixed = true;
}

void Cloth::addCollider(Collider* collider) {
    colliders.push_back(collider);
}

