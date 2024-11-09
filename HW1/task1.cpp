#include "Eigen/Eigen"
#include "Eigen/src/Core/Matrix.h"
#include "Triangle.hpp"
#include <cstddef>
#include <iostream>

// TODO: Implement this function to test if point (x, y) is inside the triangle.
/**
 * @param _v: a pointer to the array of three triangle vertices. The coordiante of
 * vertices is homogenous.
 */
static bool insideTriangle(float x, float y, const Eigen::Vector3f *_v) {
  // Lambda function to calculate the cross product of two vectors in 2D.
    auto crossProduct = [](const Eigen::Vector2f& a, const Eigen::Vector2f& b) -> float {
        return a.x() * b.y() - a.y() * b.x();
    };

    // Convert homogeneous coordinates to 2D points, assuming w=1.
    Eigen::Vector2f A = Eigen::Vector2f(_v[0].x(), _v[0].y());
    Eigen::Vector2f B = Eigen::Vector2f(_v[1].x(), _v[1].y());
    Eigen::Vector2f C = Eigen::Vector2f(_v[2].x(), _v[2].y());
    Eigen::Vector2f P = Eigen::Vector2f(x, y);

    // Compute vectors from point P to vertices A, B, and C.
    Eigen::Vector2f PA = A - P, PB = B - P, PC = C - P;

    // Calculate cross products to determine if point P is on the same side of each edge.
    float cross1 = crossProduct(B - A, PA);
    float cross2 = crossProduct(C - B, PB);
    float cross3 = crossProduct(A - C, PC);

    // Check if point P is on the inside (same sign) or outside of the triangle.
    bool has_neg = (cross1 < 0) || (cross2 < 0) || (cross3 < 0);
    bool has_pos = (cross1 > 0) || (cross2 > 0) || (cross3 > 0);

    // Point is inside the triangle if 'has_neg' is not true or 'has_pos' is not true (not both).
    return !(has_neg && has_pos);
}

int main(int argc, char const *argv[]) {
  std::vector<Triangle> triangles;
  std::vector<Vector3f> vertices{{0.f, 0.f, 1.f}, {2.f, 0.f, 1.f},
                                 {1.f, 1.f, 1.f}, {3.f, 3.f, 1.0},
                                 {3.f, 5.f, 1.f}, {5.f, 3.f, 1.f}};
  for (size_t i = 0; i < 2; ++i) {
    Triangle t;
    for (size_t j = 0; j < 3; ++j) {
      t.setVertex(j, vertices[i * 3 + j]);
    }
    triangles.push_back(t);
  }
  std::vector<Vector2f> points{{1.0, 0.5}, {4.f, 3.5f}};

  for (auto &point : points) {
    for (auto &t : triangles) {
      if (insideTriangle(point.x(), point.y(), t.v)) {
        std::cout << "Point: (" << point.x() << ", " << point.y()
                  << ") is inside the triangle: " << t << std::endl;
      }
      else{
        std::cout << "Point: (" << point.x() << ", " << point.y()
                  << ") is not inside the triangle: " << t << std::endl;
      }
    }
  }
  return 0;
}