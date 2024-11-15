#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Eigen/Dense"
// #include "Eigen/src/Core/Matrix.h"
#include "OBJ_Loader.h"
#include "Shader.hpp"
#include "Texture.hpp"
#include "Triangle.hpp"
#include "global.hpp"
#include "rasterizer.hpp"

Eigen::Matrix4f get_rotation(float rotation_angle, const Eigen::Vector3f &axis) {
    // Calculate a rotation matrix from rotation axis and angle.
    // Note: rotation_angle is in degree.
    Eigen::Matrix4f rotation_matrix = Eigen::Matrix4f::Identity();

    float rotation_angle_rad = rotation_angle * MY_PI / 180.0;
    float cos_theta = cos(rotation_angle_rad);
    float sin_theta = sin(rotation_angle_rad);

    Eigen::Vector3f axis_ = axis.normalized();
    Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f ux;
    ux << 0, -axis_.z(), axis_.y(), axis_.z(), 0, -axis_.x(), -axis_.y(), axis_.x(), 0;

    Eigen::Matrix3f rotation_matrix_3x3 =
        cos_theta * identity + (1 - cos_theta) * (axis_ * axis_.transpose()) + sin_theta * ux;
    rotation_matrix.block<3, 3>(0, 0) = rotation_matrix_3x3;

    return rotation_matrix;
}

Eigen::Matrix4f get_translation(const Eigen::Vector3f &translation) {
    // Calculate a transformation matrix of given translation vector.
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans(0, 3) = translation.x();
    trans(1, 3) = translation.y();
    trans(2, 3) = translation.z();
    return trans;
}

Eigen::Matrix4f look_at(Eigen::Vector3f eye_pos, Eigen::Vector3f target,
                        Eigen::Vector3f up = Eigen::Vector3f(0, 1, 0)) {
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Vector3f z = (eye_pos - target).normalized();
    Eigen::Vector3f x = up.cross(z).normalized();
    Eigen::Vector3f y = z.cross(x).normalized();

    Eigen::Matrix4f rotate;
    rotate << x.x(), x.y(), x.z(), 0, y.x(), y.y(), y.z(), 0, z.x(), z.y(), z.z(), 0, 0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1, -eye_pos[2], 0, 0, 0, 1;

    view = rotate * translate * view;
    return view;
}

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos) {
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    view = look_at(eye_pos, Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 1, 0));

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle, const Eigen::Vector3f &axis,
                                 const Eigen::Vector3f &translation) {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f rotation = get_rotation(rotation_angle, axis);

    Eigen::Matrix4f trans = get_translation(translation);

    model = trans * rotation * model;
    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fovy, float aspect_ratio, float zNear, float zFar) {
    // Create the projection matrix for the given parameters.
    // Then return it.
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    float eye_fovy_rad = eye_fovy * MY_PI / 180.0;
    float top = zNear * tan(eye_fovy_rad / 2.0);
    float bottom = -top;
    float right = top * aspect_ratio;
    float left = -right;

    projection << zNear / right, 0, 0, 0, 0, zNear / top, 0, 0, 0, 0, (zNear + zFar) / (zNear - zFar),
        2 * zNear * zFar / (zNear - zFar), 0, 0, -1, 0;

    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload &payload) {
    return payload.position;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f &vec, const Eigen::Vector3f &axis) {
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}





Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload &payload) {
//     Eigen::Vector3f return_color = {0, 0, 0};
//     if (payload.texture) {
//         // TODO: Get the texture value at the texture coordinates of the current fragment
//         return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
//     }
//     Eigen::Vector3f texture_color;
//     texture_color << return_color.x(), return_color.y(), return_color.z();

//     Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
//     Eigen::Vector3f kd = texture_color / 255.f;
//     Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

//     Eigen::Vector3f amb_light_intensity{10, 10, 10};

//     float p = 150;

//     std::vector<light> lights = payload.view_lights;
//     Eigen::Vector3f color = texture_color;
//     Eigen::Vector3f point = payload.view_pos;
//     Eigen::Vector3f normal = payload.normal;

//     Eigen::Vector3f result_color = {0, 0, 0};
//     Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);  // cwiseProduct--dot product

//     for (auto &light : lights) {
//         // TODO: For each light source in the code, calculate what the *ambient*,
//         // *diffuse*, and *specular* components are. Then, accumulate that result on the
//         // *result_color* object.
//         Eigen::Vector3f light_dir = (light.position - point).normalized();
//         float diff_intensity = std::max(light_dir.dot(normal), 0.0f);

//         // Diffuse component
//         Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity) * diff_intensity;

//         // Specular component
//         Eigen::Vector3f view_dir = (-point).normalized();
//         Eigen::Vector3f reflect_dir = (2.0f * normal * normal.dot(light_dir) - light_dir).normalized();
//         float spec_intensity = pow(std::max(view_dir.dot(reflect_dir), 0.0f), p);
//         Eigen::Vector3f Ls = ks.cwiseProduct(light.intensity) * spec_intensity;

//         result_color += (Ld + Ls);
        
//     }
//     result_color += La;

//     return result_color * 255.;
// }

    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture) {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f amb_light_intensity{10, 10, 10};

    float p = 150;
    
    std::vector<light> lights = payload.view_lights;
    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;
    Eigen::Vector3f eye_pos = {0, 0, 10};

    Eigen::Vector3f result_color = {0, 0, 0};
    Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);  // cwiseProduct--dot product

    for (auto &light : lights) {
        // TODO: For each light source in the code, calculate what the *ambient*,
        // *diffuse*, and *specular* components are. Then, accumulate that result on the
        // *result_color* object.

        // Eigen::Vector3f light_dir = (light.position - point).normalized();
        // float r2 = light_dir.dot(light_dir);
        // Eigen::Vector3f view_dir = (-point).normalized();
        // Eigen::Vector3f h = (light_dir + view_dir).normalized();
        
        // // Diffuse component
        // Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / r2) * std::max(0.0f, normal.normalized().dot(light_dir));

        // // Specular component
        
        // Eigen::Vector3f reflect_dir = (2.0f * normal * normal.dot(light_dir) - light_dir).normalized();
        // float spec_intensity = pow(std::max(normal.normalized().dot(h), 0.0f), p);
        // Eigen::Vector3f Ls = ks.cwiseProduct(light.intensity / r2) * spec_intensity;

        // result_color += (Ld + Ls);
        Eigen::Vector3f l = (light.position - payload.view_pos).normalized();
        Eigen::Vector3f v = (eye_pos - payload.view_pos).normalized();
        Eigen::Vector3f n = payload.normal.normalized();
        Eigen::Vector3f h = (v + l).normalized();
        Eigen::Vector3f r = (light.position - point);
        float r2 = r.dot(r);
        float diff = std::max(0.0f, n.dot(l));
        Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / r2) * diff; 
        float spec = std::pow(std::max(0.0f, n.dot(h)), p);
        Eigen::Vector3f Ls = ks.cwiseProduct(light.intensity / r2) * spec; 
        result_color += (Ld + Ls);
        
    }
    result_color += La;
    result_color = result_color.cwiseMin(1.0f).cwiseMax(0.0f);

    // return result_color * 2;
    return result_color * 255.f;
}

int main(int argc, const char **argv) {
    std::vector<Triangle *> TriangleList;

    float angle = 55.0;
    bool command_line = false;

    std::string filename = "output.png";
    rst::Shading shading = rst::Shading::Phong;
    objl::Loader Loader;
    std::string obj_path = "../models/";

    // Load .obj File
    bool loadout = Loader.LoadFile("../models/scene.obj");
    for (auto mesh : Loader.LoadedMeshes) {
        for (int i = 0; i < mesh.Vertices.size(); i += 3) {
            Triangle *t = new Triangle();
            for (int j = 0; j < 3; j++) {
                t->setVertex(j, Vector4f(mesh.Vertices[i + j].Position.X, mesh.Vertices[i + j].Position.Y,
                                         mesh.Vertices[i + j].Position.Z, 1.0));
                t->setNormal(j, Vector3f(mesh.Vertices[i + j].Normal.X, mesh.Vertices[i + j].Normal.Y,
                                         mesh.Vertices[i + j].Normal.Z));
                t->setTexCoord(
                    j, Vector2f(mesh.Vertices[i + j].TextureCoordinate.X, mesh.Vertices[i + j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "scene.png";
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = texture_fragment_shader;

    if (argc == 2) {
        command_line = true;
        filename = std::string(argv[1]);
    } 
    
    std::cout << "Rasterizing using the texture shader\n";
    active_shader = texture_fragment_shader;
    texture_path = "scene.png";
    r.set_texture(Texture(obj_path + texture_path));
    std::cout << "Rasterizing using Phong shading\n";
    shading = rst::Shading::Phong;

    Eigen::Vector3f eye_pos = {0, 10, 20};
    auto l1 = light{{-5, 20, 5}, {100, 100, 100}};
    auto l2 = light{{-20, 20, 0}, {100, 100, 100}};

    std::vector<light> lights = {l1, l2};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle, {0, 1, 0}, {0, 0, 0}));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));
        r.set_lights(lights);

        // set the shadow view
        r.set_shadow_view(get_view_matrix(l1.position));

        // calculate the dep-buf
        r.calculate_depth_map(TriangleList);

        // activate shadow
        r.draw(TriangleList, false, shading, true);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle, {0, 1, 0}, {0, 0, 0}));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));
        r.set_lights(lights);

        // set the shadow view
        r.set_shadow_view(get_view_matrix(l1.position));

        // calculate the dep-buf
        r.calculate_depth_map(TriangleList);

        // activate shadow
        r.draw(TriangleList, false, shading, true);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);
        angle += 5;
        std::cout << "frame count: " << frame_count++ << std::endl;
    }
    return 0;
}
