#include "include/rasterizer.h"
#include <iostream>
#include "include/transforms.h"

rasterizer::rasterizer(unsigned w, unsigned h) : image(h, w, CV_8UC4), width(w), height(h) {}

void rasterizer::set(unsigned int x, unsigned int y, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    if (x < width && y < height) {
        cv::Vec4b& pixel = image.at<cv::Vec4b>(y, x);
        pixel[0] = b;
        pixel[1] = g;
        pixel[2] = r;
        pixel[3] = a;
    }
}

void rasterizer::get(unsigned int x, unsigned int y, unsigned char& r, unsigned char& g, unsigned char& b, unsigned char& a) const {
    if (x < width && y < height) {
        cv::Vec4b pixel = image.at<cv::Vec4b>(y, x);
        b = pixel[0];
        g = pixel[1];
        r = pixel[2];
        a = pixel[3];
    }
}

void rasterizer::clear() {
    image.setTo(cv::Scalar(0, 0, 0, 0));
}

void rasterizer::flip_vertically() {
    cv::flip(image, image, 0);
}

cv::Mat rasterizer::to_mat() const {
    return image;
}

void rasterizer::draw_line_3D(const Eigen::Vector3f& begin, const Eigen::Vector3f& end, const Eigen::Vector3f& color) {
    Eigen::Vector4f begin_proj = projection * view * model * Eigen::Vector4f(begin.x(), begin.y(), begin.z(), 1.0f);
    Eigen::Vector4f end_proj = projection * view * model * Eigen::Vector4f(end.x(), end.y(), end.z(), 1.0f);

    begin_proj /= begin_proj.w();
    end_proj /= end_proj.w();

    begin_proj = viewport_transform(begin_proj);
    end_proj = viewport_transform(end_proj);

    line(begin_proj.x(), begin_proj.y(), end_proj.x(), end_proj.y(), color.x(), color.y(), color.z(), 255);
}


void rasterizer::line(int x0, int y0, int x1, int y1, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    bool steep = std::abs(y1 - y0) > std::abs(x1 - x0);
    if (steep) {
        std::swap(x0, y0);
        std::swap(x1, y1);
    }
    if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }

    int dx = x1 - x0;
    int dy = y1 - y0;
    int derror2 = std::abs(dy) * 2;
    int error2 = 0;
    int y = y0;

    for (int x = x0; x <= x1; x++) {
        if (steep) {
            set(y, x, r, g, b, a);
        } else {
            set(x, y, r, g, b, a);
        }
        error2 += derror2;

        if (error2 > dx) {
            y += (y1 > y0 ? 1 : -1);
            error2 -= dx * 2;
        }
    }
}


void rasterizer::drawRope3D(Rope *rope, const std::vector<unsigned char> &color) {
    if (color.size() != 3) {
        std::cerr << "Error: Invalid color vector size. It should have 3 components (RGB)." << std::endl;
        return;
    }

    unsigned char r = color[0];
    unsigned char g = color[1];
    unsigned char b = color[2];

    for (auto &s : rope->springs) {
        Eigen::Vector3f p1 = s->m1->position;
        Eigen::Vector3f p2 = s->m2->position;
        draw_line_3D(p1, p2, {r, g, b});
    }
}

void rasterizer::drawCloth(Cloth *cloth, const std::vector<unsigned char> &color) {
    if (color.size() != 3) {
        std::cerr << "Error: Invalid color vector size. It should have 3 components (RGB)." << std::endl;
        return;
    }

    if (cloth->masses.empty()) {
        std::cerr << "Error: Cloth has no masses." << std::endl;
        return;
    }

    unsigned char r = color[0];
    unsigned char g = color[1];
    unsigned char b = color[2];

    auto &masses = cloth->masses;
    int num_nodes_x = cloth->num_nodes_x;
    int num_nodes_z = cloth->num_nodes_z;

    for (int i = 0; i < num_nodes_z; ++i) {
        for (int j = 0; j < num_nodes_x - 1; ++j) {
            Eigen::Vector3f p1 = masses[i * num_nodes_x + j]->position;
            Eigen::Vector3f p2 = masses[i * num_nodes_x + j + 1]->position;
            draw_line_3D(p1, p2, {r, g, b});
        }
    }

    for (int j = 0; j < num_nodes_x; ++j) {
        for (int i = 0; i < num_nodes_z - 1; ++i) {
            Eigen::Vector3f p1 = masses[i * num_nodes_x + j]->position;
            Eigen::Vector3f p2 = masses[(i + 1) * num_nodes_x + j]->position;
            draw_line_3D(p1, p2, {r, g, b});
        }
    }
}

void rasterizer::drawBox(Box *box, const std::vector<unsigned char> &color) {
    if (color.size() != 3) {
        std::cerr << "Error: Invalid color vector size. It should have 3 components (RGB)." << std::endl;
        return;
    }

    unsigned char r = color[0];
    unsigned char g = color[1];
    unsigned char b = color[2];

    Eigen::Vector3f half_dimensions = box->dimensions / 2.0;
    Eigen::Vector3f center = box->center;

    std::vector<Eigen::Vector3f> vertices(8);
    vertices[0] = center + Eigen::Vector3f(-half_dimensions.x(), -half_dimensions.y(), -half_dimensions.z());
    vertices[1] = center + Eigen::Vector3f(half_dimensions.x(), -half_dimensions.y(), -half_dimensions.z());
    vertices[2] = center + Eigen::Vector3f(-half_dimensions.x(), half_dimensions.y(), -half_dimensions.z());
    vertices[3] = center + Eigen::Vector3f(half_dimensions.x(), half_dimensions.y(), -half_dimensions.z());
    vertices[4] = center + Eigen::Vector3f(-half_dimensions.x(), -half_dimensions.y(), half_dimensions.z());
    vertices[5] = center + Eigen::Vector3f(half_dimensions.x(), -half_dimensions.y(), half_dimensions.z());
    vertices[6] = center + Eigen::Vector3f(-half_dimensions.x(), half_dimensions.y(), half_dimensions.z());
    vertices[7] = center + Eigen::Vector3f(half_dimensions.x(), half_dimensions.y(), half_dimensions.z());

    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 5}, {2, 3},
        {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}, {6, 7}
    };

    for (const auto &edge : edges) {
        draw_line_3D(vertices[edge.first], vertices[edge.second], {r, g, b});
    }
}

void rasterizer::set_model(const Eigen::Matrix4f& m) { model = m; }

void rasterizer::set_view(const Eigen::Matrix4f& v) { view = v; }

void rasterizer::set_projection(const Eigen::Matrix4f& p) { projection = p; }

Eigen::Vector4f rasterizer::viewport_transform(const Eigen::Vector4f& v) {
    float f1 = (100 - 0.1) / 2.0;
    float f2 = (100 + 0.1) / 2.0;
    return Eigen::Vector4f(
        0.5 * width * (v.x() + 1.0),
        0.5 * height * (v.y() + 1.0),
        v.z() * f1 + f2,
        v.w()
    );
}


