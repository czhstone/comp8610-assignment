#include <iostream>
#include <vector>
#include "Eigen/Dense"

#include "include/rope.h"
#include "include/mass.h"
#include "include/spring.h"

Rope::Rope(Eigen::Vector3f start, Eigen::Vector3f end, int num_nodes, float node_mass, float k, float d, std::vector<int> fixed_nodes)
    : damping_factor(d) {
    initializeMasses(start, end, num_nodes, node_mass, fixed_nodes);
    linkSprings(k);
}

void Rope::initializeMasses(Eigen::Vector3f start, Eigen::Vector3f end, int num_nodes, float node_mass, std::vector<int> fixed_nodes) {
    // TODO: Task 1 - 
    Eigen::Vector3f interval = (end - start) / (num_nodes - 1);
    for (int i = 0; i < num_nodes; ++i) {
        Eigen::Vector3f current_position = start + i * interval;
        bool is_fixed = (std::find(fixed_nodes.begin(), fixed_nodes.end(), i) != fixed_nodes.end());
        Mass *newMass = new Mass(current_position, node_mass, is_fixed); // Allocate Mass on the heap
        masses.push_back(newMass);
    }
}

void Rope::linkSprings(float k) {

    // TODO: Task 1 - Link springs
    
    for (size_t i = 0; i < masses.size() - 1; ++i) {
        Spring *newSpring = new Spring(masses[i], masses[i + 1], k); // Allocate Spring on the heap
        springs.push_back(newSpring);
    }

}

void Rope::simulateEuler(float delta_t, Eigen::Vector3f gravity, IntegrationMethod method) {

    // TODO: Task 2 - Implement Euler integration
    
    for (auto &s : springs) {
        Eigen::Vector3f ab = s->m2->position - s->m1->position;  // Vector from m1 to m2
        float springForceMagnitude = s->k * (ab.norm() - s->rest_length);
        Eigen::Vector3f springForce = ab.normalized() * springForceMagnitude;

        s->m1->force += springForce;
        s->m2->force -= springForce;
    }

    // float damping_factor = 0.1f;  // Damping coefficient

    // Updating masses
    for (auto &m : masses) {
        if (!m->is_fixed) {
            // Add gravitational force
            m->force += gravity * m->mass;
            // Add damping force
            m->force -= damping_factor * m->velocity;

            // Calculate acceleration
            Eigen::Vector3f acceleration = m->force / m->mass;

            if (method == IntegrationMethod::EXPLICIT) {
                // Explicit Euler Integration
                m->position += m->velocity * delta_t;
                m->velocity += acceleration * delta_t;
            } else if (method == IntegrationMethod::IMPLICIT) {
                // Implicit Euler Integration
                m->velocity += acceleration * delta_t;
                m->position += m->velocity * delta_t;
            }
        }

        // Reset forces after each integration step
        m->force.setZero();
    }

}


void Rope::simulateVerlet(float delta_t, Eigen::Vector3f gravity) {

    // TODO: Task 3 - Implement Verlet integration
    // First, update all spring forces
    for (auto &s : springs) {
        Eigen::Vector3f ab = s->m2->position - s->m1->position;
        Eigen::Vector3f force = s->k * (ab.normalized()) * (ab.norm() - s->rest_length);

        s->m1->force += force;
        s->m2->force -= force;  // Apply Newton's third law
    }

    // Update positions of all masses
    for (auto &m : masses) {
        if (!m->is_fixed) {
            // Add gravity force
            m->force += gravity * m->mass;

            // Calculate the acceleration from the resultant force
            Eigen::Vector3f acceleration = m->force / m->mass;

            // Store the current position to update the last position later
            Eigen::Vector3f temp_position = m->position;

            // Verlet integration formula with damping for position update
            // double damping_factor = 0.00005; // Smaller damping factor suitable for Verlet stability
            m->position = m->position + (1 - damping_factor) * (m->position - m->last_position) + acceleration * delta_t * delta_t;

            // Update last_position to the stored current position for the next timestep
            m->last_position = temp_position;
        }

        // Reset forces after calculations for the next timestep
        m->force = Eigen::Vector3f(0, 0, 0);
    }
        
}