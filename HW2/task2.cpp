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

// TODO: Task2 Implement the following fragment shaders
Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload &payload) {
    // Eigen::Vector3f result;
    
    // // convert nomral vector from [-1, 1] to [0, 1] and then to [0, 255]
    // return result;

    Eigen::Vector3f n = payload.normal.normalized();
    // Map from [-1, 1] to [0, 1]
    Eigen::Vector3f result = (n.array() + 1.0f) * 0.5f;
    // Then to [0, 255] for RGB
    return result * 255.f;
}

// TODO: Task2 Implement the following fragment shaders
Eigen::Vector3f blinn_phong_fragment_shader(const fragment_shader_payload &payload) {
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f amb_light_intensity{10, 10, 10};

    float p = 150;

    auto lights = payload.view_lights;
    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;
    Eigen::Vector3f result_color = {0, 0, 0};

    Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);  // cwiseProduct--dot product
    for (auto &light : lights) {
        // TODO: For each light source in the code, calculate what the *diffuse*, and
        // *specular* components are. Then, accumulate that result on the *result_color*
        // object.
        Eigen::Vector3f light_dir = (light.position - payload.view_pos).normalized();
        Eigen::Vector3f view_dir = (-payload.view_pos).normalized();
        Eigen::Vector3f reflect_dir = reflect(-light_dir, payload.normal).normalized();
        float diff = std::max(payload.normal.dot(light_dir), 0.0f);
        Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity / (light.position - payload.view_pos).squaredNorm()) * diff;
        float spec = std::pow(std::max(-view_dir.dot(reflect_dir), 0.0f), p);
        Eigen::Vector3f specular = ks.cwiseProduct(light.intensity / (light.position - payload.view_pos).squaredNorm()) * spec;

        result_color += diffuse + specular;
    }
    result_color += La;

    return result_color * 255.f * 2;
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

    Eigen::Vector3f result_color = {0, 0, 0};
    Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);  // cwiseProduct--dot product

    for (auto &light : lights) {
        // TODO: For each light source in the code, calculate what the *ambient*,
        // *diffuse*, and *specular* components are. Then, accumulate that result on the
        // *result_color* object.
        Eigen::Vector3f light_dir = (light.position - point).normalized();
        float r2 = light_dir.dot(light_dir);
        Eigen::Vector3f view_dir = (-point).normalized();
        Eigen::Vector3f h = (light_dir + view_dir).normalized();
        
        // Diffuse component
        Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / r2) * std::max(0.0f, normal.normalized().dot(light_dir));

        // Specular component
        
        Eigen::Vector3f reflect_dir = (2.0f * normal * normal.dot(light_dir) - light_dir).normalized();
        float spec_intensity = pow(std::max(normal.normalized().dot(h), 0.0f), p);
        Eigen::Vector3f Ls = ks.cwiseProduct(light.intensity / r2) * spec_intensity;

        result_color += (Ld + Ls);
        
    }
    result_color += La;

    return result_color * 2;
}


// TODO: Task2 Implement the following fragment shaders
Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload &payload) {
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f amb_light_intensity{10, 10, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;
    float kh = 0.2, kn = 0.1;

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (f(u+1/w,v)-f(u,v))
    // dV = kh * kn * (f(u,v+1/h)-f(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)
    // You can implement the function f as f(u,v) = norm of payload.texture->getColor(u, v) 
    Eigen::Vector3f result_color = {0, 0, 0};

    // Assuming payload has texture coordinates (u, v) and a method to fetch texture color
    float u = payload.tex_coords.x();
    float v = payload.tex_coords.y();
    float w = payload.texture->width;
    float h = payload.texture->height;

    auto f = [&](float u, float v) -> float {
        // Placeholder for the norm of the color vector from texture
        Eigen::Vector3f color = payload.texture->getColor(u, v);
        return color.norm();
    };

    // Calculating tangent and bitangent
    Eigen::Vector3f t(normal.x() * normal.y() / sqrt(normal.x() * normal.x() + normal.z() * normal.z()), sqrt(normal.x() * normal.x() + normal.z() * normal.z()), normal.z() * normal.y() / sqrt(normal.x() * normal.x() + normal.z() * normal.z()));
    Eigen::Vector3f b = normal.cross(t);

    // TBN matrix
    Eigen::Matrix3f TBN;
    TBN.col(0) = t.normalized();
    TBN.col(1) = b.normalized();
    TBN.col(2) = normal.normalized();

    // Derivatives dU and dV
    float dU = kh * kn * (f(u + 1.0f / w, v) - f(u, v));
    float dV = kh * kn * (f(u, v + 1.0f / h) - f(u, v));

    // Adjusted normal vector
    Eigen::Vector3f ln(-dU, -dV, 1.0f);
    Eigen::Vector3f n = (TBN * ln).normalized();

    Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);  // cwiseProduct--dot product
    std::vector<light> lights = payload.view_lights;

    // for (auto &light : lights) {
    //     // TODO: For each light source in the code, calculate what the *diffuse*, and
    //     // *specular* components are. Then, accumulate that result on the *result_color*
    //     // object.
    //     Eigen::Vector3f light_dir = (light.position - payload.view_pos).normalized();
    //     Eigen::Vector3f view_dir = (-payload.view_pos).normalized();
    //     Eigen::Vector3f reflect_dir = reflect(-light_dir, n).normalized();
    //     float diff = std::max(n.dot(light_dir), 0.0f);
    //     Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity / (light.position - payload.view_pos).squaredNorm()) * diff;
    //     float spec = std::pow(std::max(-view_dir.dot(reflect_dir), 0.0f), p);
    //     Eigen::Vector3f specular = ks.cwiseProduct(light.intensity / (light.position - payload.view_pos).squaredNorm()) * spec;

    //     result_color += diffuse + specular;
        
    // }
    // result_color += La;
    result_color = n * 255.;

    

    return result_color ;
}

int main(int argc, const char **argv) {
    std::vector<Triangle *> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    rst::Shading shading = rst::Shading::Phong;
    objl::Loader Loader;
    std::string obj_path = "../models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");
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

    auto texture_path = "hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = blinn_phong_fragment_shader;

    if (argc < 3) {
        std::cout << "Usage: [Shader (texture, normal, blinn-phong, bump)] [Shading "
                     "Frequency (Flat, Gouraud, Phong)]  [savename.png]"
                  << std::endl;
        return 1;
    } else {
        if (argc == 4) {
            command_line = true;
            filename = std::string(argv[3]);
        }
        if (std::string(argv[1]) == "texture") {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        } else if (std::string(argv[1]) == "normal") {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        } else if (std::string(argv[1]) == "blinn-phong") {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = blinn_phong_fragment_shader;
        } else if (std::string(argv[1]) == "bump") {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }

        if (std::string(argv[2]) == "Flat") {
            std::cout << "Rasterizing using Flat shading\n";
            shading = rst::Shading::Flat;
        } else if (std::string(argv[2]) == "Gouraud") {
            std::cout << "Rasterizing using Goround shading\n";
            shading = rst::Shading::Gouraud;
        } else if (std::string(argv[2]) == "Phong") {
            std::cout << "Rasterizing using Phong shading\n";
            shading = rst::Shading::Phong;
        }
    }

    Eigen::Vector3f eye_pos = {0, 0, 10};
    auto l1 = light{{-5, 5, 5}, {50, 50, 50}};
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

        r.draw(TriangleList, true, shading);
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

        r.draw(TriangleList, true, shading);
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
