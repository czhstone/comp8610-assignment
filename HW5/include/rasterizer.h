#ifndef rasterizer_H
#define rasterizer_H

#include <string>
#include <opencv2/core.hpp>
#include <Eigen/Eigen>
#include "Eigen/src/Core/Matrix.h"
#include "rope.h"
#include "cloth.h"
#include "box.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class rasterizer {
private:
    cv::Mat image;

    Eigen::Matrix4f model;
    Eigen::Matrix4f view;
    Eigen::Matrix4f projection;


public:
    unsigned width, height;
    rasterizer(unsigned w, unsigned h);

    void set_model(const Eigen::Matrix4f &m);
    void set_view(const Eigen::Matrix4f &v);
    void set_projection(const Eigen::Matrix4f &p);
    Eigen::Vector4f viewport_transform(const Eigen::Vector4f& v);



    void set(unsigned int x, unsigned int y, unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255);
    void get(unsigned int x, unsigned int y, unsigned char& r, unsigned char& g, unsigned char& b, unsigned char& a) const;
    void flip_vertically();
    void clear();
    const unsigned char* buffer() const;
    cv::Mat to_mat() const;


    void line(int x0, int y0, int x1, int y1, unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255);

    void draw_line_3D(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& color);

    void drawRope3D(Rope *rope, const std::vector<unsigned char> &color);

    void drawCloth(Cloth *cloth, const std::vector<unsigned char> &color);

    void drawBox(Box *box, const std::vector<unsigned char> &color);
};

#endif // rasterizer_H
