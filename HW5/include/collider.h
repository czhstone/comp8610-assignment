#ifndef COLLIDER_H
#define COLLIDER_H

#include <Eigen/Dense>
#include "mass.h"

class Collider {
public:
    virtual ~Collider() {}
    virtual bool checkCollision(Mass& mass) const = 0;
    virtual void resolveCollision(Mass& mass) const = 0;
};

#endif // COLLIDER_H
