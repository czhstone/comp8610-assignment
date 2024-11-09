#ifndef SPRING_H
#define SPRING_H

#include "mass.h"

using namespace std;

struct Spring {
    Spring(Mass* a, Mass* b, float k)
        : m1(a), m2(b), k(k), rest_length((a->position - b->position).norm()) {}

    Mass *m1;
    Mass *m2;

    float k;
    double rest_length;

};


#endif /* SPRING_H */
