
#include "rasterizer.hpp"

#include <math.h>

#include <algorithm>
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

#include "Shader.hpp"
#include "Triangle.hpp"

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions) {
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices) {
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols) {
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f> &normals) {
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}

void rst::rasterizer::post_process_buffer() {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = get_index(x, y);
            for (int i = 0; i < 4; i++) {
                frame_buf[index] += ssaa_frame_buf[4 * index + i];
            }
            frame_buf[index] /= 4;
        }
    }
}

// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end) {
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

    dx = x2 - x1;
    dy = y2 - y1;
    dx1 = fabs(dx);
    dy1 = fabs(dy);
    px = 2 * dy1 - dx1;
    py = 2 * dx1 - dy1;

    if (dy1 <= dx1) {
        if (dx >= 0) {
            x = x1;
            y = y1;
            xe = x2;
        } else {
            x = x2;
            y = y2;
            xe = x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; x < xe; i++) {
            x = x + 1;
            if (px < 0) {
                px = px + 2 * dy1;
            } else {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
                    y = y + 1;
                } else {
                    y = y - 1;
                }
                px = px + 2 * (dy1 - dx1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    } else {
        if (dy >= 0) {
            x = x1;
            y = y1;
            ye = y2;
        } else {
            x = x2;
            y = y2;
            ye = y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; y < ye; i++) {
            y = y + 1;
            if (py <= 0) {
                py = py + 2 * dx1;
            } else {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
                    x = x + 1;
                } else {
                    x = x - 1;
                }
                py = py + 2 * (dx1 - dy1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f) {
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(float x, float y, const Vector4f *_v) {
    Vector3f v[3];
    for (int i = 0; i < 3; i++)
        v[i] = {_v[i].x(), _v[i].y(), 1.0};
    Vector3f p(x, y, 1.);
    Vector3f f0, f1, f2;
    f0 = (p - v[0]).cross(v[1] - v[0]);
    f1 = (p - v[1]).cross(v[2] - v[1]);
    f2 = (p - v[2]).cross(v[0] - v[2]);
    if (f0.dot(f1) > 0 && f1.dot(f2) > 0)
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f *v) {
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) /
               (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() -
                v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) /
               (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() -
                v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) /
               (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() -
                v[1].x() * v[0].y());
    return {c1, c2, c3};
}

// TODO: Task1 Implement this function
void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type, bool culling, bool anti_aliasing) {
    auto &buf = pos_buf[pos_buffer.pos_id];
    auto &ind = ind_buf[ind_buffer.ind_id];
    auto &col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto &i : ind) {
        Triangle t;

        std::array<Eigen::Vector4f, 3> mm{view * model * to_vec4(buf[i[0]], 1.0f),
                                          view * model * to_vec4(buf[i[1]], 1.0f),
                                          view * model * to_vec4(buf[i[2]], 1.0f)};

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v) { return v.template head<3>(); });
        
        // TODO: Task1 Enable back face culling

        if (culling) {
            // Calculate the normal of the triangle in view space
            Eigen::Vector3f v0 = viewspace_pos[0];
            Eigen::Vector3f v1 = viewspace_pos[1];
            Eigen::Vector3f v2 = viewspace_pos[2];
            Eigen::Vector3f normal = (v1 - v0).cross(v2 - v0).normalized();

            // View direction is assumed to be from the origin since the camera is looking towards -z in view space
            Eigen::Vector3f view_dir = v0.normalized(); // Using v0 as an arbitrary point on the triangle to determine view direction

        // Back-face culling
        if (normal.dot(view_dir) >= 0) {
            // The triangle is facing away from the camera, so we skip drawing it
            continue;
        }
        }

        Eigen::Vector4f v[] = {mvp * to_vec4(buf[i[0]], 1.0f), mvp * to_vec4(buf[i[1]], 1.0f),
                               mvp * to_vec4(buf[i[2]], 1.0f)};

        // Homogeneous division
        for (auto &vec : v) {
            vec /= vec.w();
        }
        // Viewport transformation
        for (auto &vert : v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i) {
            t.setVertex(i, v[i]);
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);
/*************** update below *****************/
        rasterize_triangle(t, anti_aliasing);
    }
    if (anti_aliasing){
        post_process_buffer();
    }
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList, bool culling, rst::Shading shading, bool shadow) {
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;

    std::vector<light> viewspace_lights;
    for (const auto &l : lights) {
        light view_space_light;
        view_space_light.position = (view * to_vec4(l.position, 1.0f)).head<3>();
        view_space_light.intensity = l.intensity;
        viewspace_lights.push_back(view_space_light);
    }

    for (const auto &t : TriangleList) {
        Triangle newtri = *t;

        std::array<Eigen::Vector4f, 3> mm{(view * model * t->v[0]), (view * model * t->v[1]), (view * model * t->v[2])};

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v) { return v.template head<3>(); });

        //TODO: Task1 Enable back face culling

        if (culling) {
            // Calculate the normal of the triangle in view space
            Eigen::Vector3f v0 = viewspace_pos[0];
            Eigen::Vector3f v1 = viewspace_pos[1];
            Eigen::Vector3f v2 = viewspace_pos[2];
            Eigen::Vector3f normal = (v1 - v0).cross(v2 - v0).normalized();

            // View direction is assumed to be from the origin since the camera is looking towards -z in view space
            Eigen::Vector3f view_dir = v0.normalized(); // Using v0 as an arbitrary point on the triangle to determine view direction

        // Back-face culling
        if (normal.dot(view_dir) >= 0) {
            // The triangle is facing away from the camera, so we skip drawing it
            continue;
        }

        }

        Eigen::Vector4f v[] = {mvp * t->v[0], mvp * t->v[1], mvp * t->v[2]};
        // Homogeneous division
        for (auto &vec : v) {
            vec.x() /= vec.w();
            vec.y() /= vec.w();
            vec.z() /= vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {inv_trans * to_vec4(t->normal[0], 0.0f), inv_trans * to_vec4(t->normal[1], 0.0f),
                               inv_trans * to_vec4(t->normal[2], 0.0f)};

        // Viewport transformation
        for (auto &vert : v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i) {
            // screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i) {
            // view space normal
            newtri.setNormal(i, n[i].head<3>());
        }

        newtri.setColor(0, 148, 121.0, 92.0);
        newtri.setColor(1, 148, 121.0, 92.0);
        newtri.setColor(2, 148, 121.0, 92.0);

        // Also pass view space vertice position
        rasterize_triangle(newtri, viewspace_pos, viewspace_lights, shading, shadow);
    }
}


static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f &vert1,
                                   const Eigen::Vector3f &vert2, const Eigen::Vector3f &vert3, float weight) {
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f &vert1,
                                   const Eigen::Vector2f &vert2, const Eigen::Vector2f &vert3, float weight) {
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}

// TODO: Task1 Implement this function
void rst::rasterizer::rasterize_triangle(const Triangle &t, bool anti_aliasing) {
    auto v = t.toVector4();

    // 1. Find out the bounding box of current triangle.
    // 2. Iterate through the pixel and find if the current pixel is inside the triangle
    // Subpixel sampling if do anti-aliasing
    // 3. If so, use the following code to get the interpolated z value.
    // auto[alpha, beta, gamma] = computeBarycentric2D(x+0.5, y+0.5, t.v);
    // float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    // float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    // z_interpolated *= w_reciprocal;
    // 4. Check if the interpolated depth is closer than the value in the depth buffer.
    // 5. If so, update the depth buffer and set the current pixel to the color of the interpolated color of vertices
    // set_pixel(Vector2i(x, y), interpolated_color);
    // 1. Calculate the bounding box
    float minX = std::min({v[0].x(), v[1].x(), v[2].x()});
    float maxX = std::max({v[0].x(), v[1].x(), v[2].x()});
    float minY = std::min({v[0].y(), v[1].y(), v[2].y()});
    float maxY = std::max({v[0].y(), v[1].y(), v[2].y()});


    // Convert to pixel coordinates
    minX = floor(minX); maxX = ceil(maxX);
    minY = floor(minY); maxY = ceil(maxY);

    if (anti_aliasing) 
    {
        // super-sampling method
        for (int x = minX * 2; x <= maxX * 2; x++) {
            for (int y = minY * 2; y <= maxY * 2; y++) {
                float subX = x / 2.0f;
                float subY = y / 2.0f;
                if (insideTriangle(subX, subY, v.data())) {
                    auto [alpha, beta, gamma] = computeBarycentric2D(subX, subY, v.data());
                    float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
                    // One of the four oversampled pixels corresponding to the original pixel
                    int ssaa_index = get_index(x / 2, y / 2) * 4 + (x % 2) + (y % 2) * 2;
                    if (ssaa_depth_buf[ssaa_index] > z_interpolated) {
                        ssaa_depth_buf[ssaa_index] = z_interpolated;
                        Eigen::Vector3f color = alpha * t.color[0] + beta * t.color[1] + gamma * t.color[2];
                        // set the color
                        ssaa_frame_buf[ssaa_index] = color * 255.;
                    }
                }
            }
        }
    }
    else {
            // 2. Iterate through each pixel in the bounding box
        for (int x = minX; x <= maxX; x++) {
            for (int y = minY; y <= maxY; y++) {
                // Consider the center of the pixel to determine if it's inside the triangle.
                float centerX = x + 0.5f, centerY = y + 0.5f;

                // Check if the pixel center is inside the triangle
                if (insideTriangle(centerX, centerY, v.data())) {
                    auto [alpha, beta, gamma] = computeBarycentric2D(centerX, centerY, v.data());

                    float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() +
                                        beta * v[1].z() / v[1].w() +
                                        gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    // 4. Depth test
                    if (depth_buf[get_index(x, y)] > z_interpolated) {
                        depth_buf[get_index(x, y)] = z_interpolated;

                        // Interpolate color based on vertex colors and barycentric coordinates
                        Eigen::Vector3f color = alpha * t.color[0] + beta * t.color[1] + gamma * t.color[2];
                        
                        // Set the pixel
                        set_pixel(Eigen::Vector2i(x, y), color * 255.);
                    }
                }
            }
        }
    }
}

// TODO: Task2 Implement this function
void rst::rasterizer::rasterize_triangle(const Triangle &t, const std::array<Eigen::Vector3f, 3> &view_pos,
                                         const std::vector<light> &view_lights, rst::Shading shading, bool shadow) {
    auto v = t.toVector4();

    if (shading == rst::Shading::Flat) {

        // 1. Find the bounding box of the triangle
        float minX = std::floor(std::min({v[0].x(), v[1].x(), v[2].x()}));
        float maxX = std::ceil(std::max({v[0].x(), v[1].x(), v[2].x()}));
        float minY = std::floor(std::min({v[0].y(), v[1].y(), v[2].y()}));
        float maxY = std::ceil(std::max({v[0].y(), v[1].y(), v[2].y()}));

        // Iterate through each pixel within the bounding box
        for (int x = minX; x <= maxX; ++x) {
            for (int y = minY; y <= maxY; ++y) {
                float x_pos = x + 0.5f;
                float y_pos = y + 0.5f;

                // Check if the pixel is inside the triangle
                if (insideTriangle(x_pos, y_pos, v.data())) {
                    auto [alpha, beta, gamma] = computeBarycentric2D(x_pos, y_pos, v.data());
                    float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = (alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w()) * w_reciprocal;

                    // Depth test
                    if (depth_buf[get_index(x, y)] > z_interpolated) {
                        // Prepare fragment shader payload
                        fragment_shader_payload payload(t.color[0], t.normal[0], t.tex_coords[0], view_lights, texture ? &*texture : nullptr);
                        payload.view_pos = view_pos[0];
                        
                        // Calculate pixel color using fragment shader
                        Eigen::Vector3f color = fragment_shader(payload);

                        // Update depth buffer and set the pixel color
                        depth_buf[get_index(x, y)] = z_interpolated;
                        set_pixel(Eigen::Vector2i(x, y), color);
                    }
                }
            }
        }
    } else if (shading == rst::Shading::Gouraud) {
            std::array<Eigen::Vector3f, 3> VertexColors;


            for (int i = 0; i < 3; ++i) {
                fragment_shader_payload payload(t.color[i], t.normal[i], t.tex_coords[i], view_lights, texture ? &*texture : nullptr);
                payload.view_pos = view_pos[i];
                VertexColors[i] = fragment_shader(payload);
            }

        float minX = std::floor(std::min({v[0].x(), v[1].x(), v[2].x()}));
        float maxX = std::ceil(std::max({v[0].x(), v[1].x(), v[2].x()}));
        float minY = std::floor(std::min({v[0].y(), v[1].y(), v[2].y()}));
        float maxY = std::ceil(std::max({v[0].y(), v[1].y(), v[2].y()}));

        // 在三角形的每个像素上进行颜色插值
        // 注意：此部分与上面的Phong着色中的迭代像素循环相似，但是这里我们插值的是顶点颜色而非法线或纹理坐标
        for (int x = minX; x <= maxX; ++x) {
            for (int y = minY; y <= maxY; ++y) {
                float x_pos = x + 0.5f;
                float y_pos = y + 0.5f;

                if (insideTriangle(x_pos, y_pos, v.data())) {
                    auto [alpha, beta, gamma] = computeBarycentric2D(x_pos, y_pos, v.data());
                    float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = (alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w()) * w_reciprocal;

                    // Depth test
                    if (depth_buf[get_index(x, y)] > z_interpolated) {
                        Eigen::Vector3f gouraudColor = alpha * VertexColors[0] + beta * VertexColors[1] + gamma * VertexColors[2];

                        // 更新深度缓冲区并设置像素颜色
                        depth_buf[get_index(x, y)] = z_interpolated;
                        set_pixel(Eigen::Vector2i(x, y), gouraudColor);
                    }
                }
            }
        }
    } else if (shading == rst::Shading::Phong) {
        // Find the bounding box of the triangle.

        // iterate through the pixel and find if the current pixel is inside the
        // triangle If so, use the following code to get the interpolated z value.
        // auto[alpha, beta, gamma] = computeBarycentric2D(x+0.5, y+0.5, t.v);
        // float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma /v[2].w());
        // float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() /
        // v[2].w(); z_interpolated *= w_reciprocal;

        // Check if the interpolated depth is closer than the value in the depth buffer.
        // If so, update the depth buffer
        // and calculate interpolated_color, interpolated_normal, interpolated_texcoords and interpolated_shadingcoords
        // float weight = alpha + beta + gamma;

        // pass them to the fragment_shader_payload
        // fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(), interpolated_texcoords,
        // view_lights, texture ? &*texture : nullptr); payload.view_pos = interpolated_shadingcoords;

        // Call the fragment shader to get the pixel color
        // auto pixel_color = fragment_shader(payload);

        // modify the color if do shadow mapping

        // 1. Find the bounding box of the triangle
        float minX = std::floor(std::min({v[0].x(), v[1].x(), v[2].x()}));
        float maxX = std::ceil(std::max({v[0].x(), v[1].x(), v[2].x()}));
        float minY = std::floor(std::min({v[0].y(), v[1].y(), v[2].y()}));
        float maxY = std::ceil(std::max({v[0].y(), v[1].y(), v[2].y()}));

        // Iterate through each pixel within the bounding box
        for (int x = minX; x <= maxX; ++x) {
            for (int y = minY; y <= maxY; ++y) {
                float x_pos = x + 0.5f;
                float y_pos = y + 0.5f;

                // Check if the pixel is inside the triangle
                if (insideTriangle(x_pos, y_pos, v.data())) {
                    auto [alpha, beta, gamma] = computeBarycentric2D(x_pos, y_pos, v.data());
                    float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = (alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w()) * w_reciprocal;

                    // Depth test
                    if (depth_buf[get_index(x, y)] > z_interpolated) {
                        // update the depth color 
                        depth_buf[get_index(x, y)] = z_interpolated;
                        // Calculate interpolated attributes for Phong shading
                        Eigen::Vector3f interpolated_color = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2], 1);
                        Eigen::Vector3f interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], 1).normalized();
                        Eigen::Vector2f interpolated_texcoords = interpolate(alpha, beta, gamma, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], 1);
                        Eigen::Vector3f interpolated_pos = interpolate(alpha, beta, gamma, view_pos[0], view_pos[1], view_pos[2], 1);
                        if (shadow){
                            Eigen::Vector4f pos_world = (view.inverse() * Eigen::Vector4f(interpolated_pos.x(), 
                                interpolated_pos.y(), interpolated_pos.z(), w_reciprocal));
                            Eigen::Vector4f pos_shadow_view = shadow_view * pos_world;
                            Eigen::Vector4f pos_shadow_image = projection * pos_shadow_view;
                            pos_shadow_image /= pos_shadow_image.w();

                            //
                            auto shadow_x = 0.5 * width * (pos_shadow_image.x() + 1.0);
                            auto shadow_y = 0.5 * height * (pos_shadow_image.y() + 1.0);
                            auto shadow_z = pos_shadow_image.z() * (50.0 - 0.1) / 2.0 + (50.0 + 0.1) / 2.0;
                            shadow_x = std::min(std::max(int(shadow_x), 0), width - 1);
                            shadow_y = std::min(std::max(int(shadow_y), 0), height - 1);      
                            int shadow_idx = get_index(shadow_x, shadow_y);
                            bool in_shadow = shadow_z > (shadow_buf[shadow_idx]+1e-2);
                            fragment_shader_payload payload(interpolated_color, interpolated_normal, interpolated_texcoords,
                             view_lights, texture ? &*texture : nullptr);
                            payload.view_pos = interpolated_pos; 
                            auto pixel_color = fragment_shader(payload);
                            set_pixel(Eigen::Vector2i(x, y), in_shadow ? pixel_color * 0.2 : pixel_color);

                        }
                        else {
                            // Prepare fragment shader payload
                        fragment_shader_payload payload;
                        payload.color = interpolated_color;
                        payload.normal = interpolated_normal;
                        payload.tex_coords = interpolated_texcoords;
                        payload.view_pos = interpolated_pos; // position in view space for lighting calculations
                        payload.view_lights = view_lights;
                        payload.texture = texture ? &*texture : nullptr;

                        // Calculate pixel color using fragment shader
                        Eigen::Vector3f pixel_color = fragment_shader(payload);

                        // Set the pixel color
                        set_pixel(Eigen::Vector2i(x, y), pixel_color);
                        }
                        
                    }
                }
            }
        }
        // set the pixel color to the frame buffer.
        // set_pixel(Vector2i(x, y), pixel_color);
    }
}

void rst::rasterizer::calculate_depth_map(const std::vector<Triangle*>& triangles) {
    Eigen::Matrix4f light_mvp = projection* shadow_view * model;
    for (const auto& triangle : triangles) {
        Triangle transformed_triangle = *triangle; 
        for (int i = 0; i < 3; ++i) {
            Eigen::Vector4f transformed_vertex = light_mvp * triangle->v[i];
            transformed_vertex /= transformed_vertex.w(); 
            //viewport
            transformed_vertex.x() = 0.5 * width * (transformed_vertex.x() + 1.0);
            transformed_vertex.y() = 0.5 * height * (transformed_vertex.y() + 1.0);
            transformed_vertex.z() = transformed_vertex.z() * (50.0 - 0.1) / 2.0 + (50.0 + 0.1) / 2.0;
            transformed_triangle.setVertex(i, transformed_vertex);
        }
        rasterize_triangle_depth(transformed_triangle);
    
    }
}

// help function to represent depth-buffer rasterization of a triangle
void rst::rasterizer::rasterize_triangle_depth(const Triangle& t) {
    auto v = t.toVector4();
    // Rasterize the triangle and update depth values in depth_buffer
    // ...
    float minX = std::min({v[0].x(), v[1].x(), v[2].x()});
    float maxX = std::max({v[0].x(), v[1].x(), v[2].x()});
    float minY = std::min({v[0].y(), v[1].y(), v[2].y()});
    float maxY = std::max({v[0].y(), v[1].y(), v[2].y()});
    minX = std::max(minX, 0.0f);
    maxX = std::min(maxX, static_cast<float>(width - 1));
    minY = std::max(minY, 0.0f);
    maxY = std::min(maxY, static_cast<float>(height - 1));
    //iterate every pixels to judge if it inside triangle
    for (int x = static_cast<int>(minX); x <= maxX; ++x) {
        for (int y = static_cast<int>(minY); y <= maxY; ++y) {
            float x_center = x + 0.5f;
            float y_center = y + 0.5f;
            if(insideTriangle(x_center,y_center,v.data())){
                auto[alpha, beta, gamma] = computeBarycentric2D(x_center, y_center, v.data());
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                int idx = get_index(x, y);
                if(z_interpolated < shadow_buf[idx]) {
                    //z_interpolated = (z_interpolated - 48.0471) / (49.73 - 48.0471);
                    shadow_buf[idx] = z_interpolated;
                    
                }
            }
        }
    }  
}



void rst::rasterizer::set_model(const Eigen::Matrix4f &m) {
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f &v) {
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p) {
    projection = p;
}

void rst::rasterizer::set_lights(const std::vector<light> &l) {
    lights = l;
}

void rst::rasterizer::set_shadow_view(const Eigen::Matrix4f &v) {
    shadow_view = v;
}

void rst::rasterizer::set_shadow_buffer(const std::vector<float> &shadow_buffer) {
    std::copy(shadow_buffer.begin(), shadow_buffer.end(), this->shadow_buf.begin());
}

void rst::rasterizer::clear(rst::Buffers buff) {
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(ssaa_frame_buf.begin(), ssaa_frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(ssaa_depth_buf.begin(), ssaa_depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(shadow_buf.begin(), shadow_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    shadow_buf.resize(w * h);
    ssaa_frame_buf.resize(4 * w * h);
    ssaa_depth_buf.resize(4 * w * h);
    texture = std::nullopt;
}

int rst::rasterizer::get_index(int x, int y) {
    return (height - y - 1) * width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color) {
    // old index: auto ind = point.y() + point.x() * width;
    int ind = (height - point.y() - 1) * width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader) {
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader) {
    fragment_shader = frag_shader;
}
