#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <iostream>
#include <opencv2/core/base.hpp>
#include <opencv2/opencv.hpp>

#include "Triangle.hpp"
#include "mesh.hpp"
#include "rasterizer.hpp"

constexpr double MY_PI = 3.1415926;

// TODO: Copy from task3
Eigen::Matrix4f get_translation(const Eigen::Vector3f &translation) {
  // Calculate a transformation matrix of given translation vector.
  Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
  trans(0, 3) = translation.x();
  trans(1, 3) = translation.y();
  trans(2, 3) = translation.z();

  return trans;
}

// TODO: Copy from task3
/**
 * @brief Get the rotation transformation matrix given rotation angle and axis.
 *
 * @param rotation_angle: rotation angle in degree.
 * @param axis: rotation axis.
 */
Eigen::Matrix4f get_rotation(float rotation_angle, const Eigen::Vector3f &axis) {
  Eigen::Matrix3f rotation_mat3 = Eigen::AngleAxisf(rotation_angle * MY_PI / 180.0f, axis.normalized()).toRotationMatrix();
  Eigen::Matrix4f rotation_matrix = Eigen::Matrix4f::Identity();
  rotation_matrix.block<3,3>(0, 0) = rotation_mat3;

  return rotation_matrix;
}

// TODO: Implement this function
Eigen::Matrix4f get_scaling(const Eigen::Vector3f &scaling) {
  Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
  scale(0, 0) = scaling.x();
  scale(1, 1) = scaling.y();
  scale(2, 2) = scaling.z();
  return scale;
}

// TODO: Copy from task3
/**
 * @brief Get the view matrix by given eye position, target view position and up vector.
 *
 * @param eye_pos: location of the camera
 * @param target: the point the camera is looking at
 */
Eigen::Matrix4f look_at(Eigen::Vector3f eye_pos, Eigen::Vector3f target,
                        Eigen::Vector3f up = Eigen::Vector3f(0, 1, 0)) {
    // forward
    Eigen::Vector3f forward = (target - eye_pos).normalized();
    
    // right
    Eigen::Vector3f right = forward.cross(up).normalized();
    
    // up
    Eigen::Vector3f new_up = right.cross(forward);

    // view matrix
    Eigen::Matrix4f view;
    view << right.x(),    right.y(),    right.z(),    -right.dot(eye_pos),
            new_up.x(),   new_up.y(),   new_up.z(),   -new_up.dot(eye_pos),
            -forward.x(), -forward.y(), -forward.z(), forward.dot(eye_pos),
            0,            0,            0,            1;
    
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
  Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

  // Perspective to orthographic projection
  Eigen::Matrix4f persp_to_ortho = Eigen::Matrix4f::Identity();
  persp_to_ortho << zNear, 0, 0, 0, 0, zNear, 0, 0, 0, 0, -(zNear + zFar), -zNear * zFar, 0, 0, -1,
      0;

  // translate to origin
  float eye_fovy_rad = eye_fovy * MY_PI / 180.0;
  float top = zNear * tan(eye_fovy_rad / 2.0);
  float bottom = -top;
  float right = top * aspect_ratio;
  float left = -right;

  Eigen::Matrix4f ortho_translate = Eigen::Matrix4f::Identity();
  ortho_translate << 1, 0, 0, -(left + right) / 2., 0, 1, 0, -(bottom + top) / 2., 0, 0, 0,
      -(zNear + zFar) / 2., 0, 0, 0, 1;

  // scale to NDC
  Eigen::Matrix4f ortho_scale = Eigen::Matrix4f::Identity();
  ortho_scale << 2. / (right - left), 0, 0, 0, 0, 2. / (top - bottom), 0, 0, 0, 0,
      2. / (zFar - zNear), 0, 0, 0, 0, 1;

  projection = ortho_scale * ortho_translate * persp_to_ortho * projection;

  return projection;
}

// TODO: Implement this function.
Mesh create_dog_mesh(const Mesh &cube) {
  Mesh dog;

  Eigen::Matrix4f translation_matrix1 = get_translation(Eigen::Vector3f(-1, 0, 0)); 
  Eigen::Matrix4f rotation_matrix1 = get_rotation(0, Eigen::Vector3f(0, 1, 0));    
  Eigen::Matrix4f scaling_matrix1 = get_scaling(Eigen::Vector3f(2, 0.6, 0.5));  
  Eigen::Matrix4f combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh body1 = cube.transform(combined_transform1);

  Eigen::Matrix4f translation_matrix2 = get_translation(Eigen::Vector3f(0, 0, 0)); 
  Eigen::Matrix4f rotation_matrix2 = get_rotation(-15, Eigen::Vector3f(0, 0, 1));    
  Eigen::Matrix4f scaling_matrix2 = get_scaling(Eigen::Vector3f(1.5, 1, 1));  
  Eigen::Matrix4f combined_transform2 = translation_matrix2 * rotation_matrix2 * scaling_matrix2;

  Mesh body2 = cube.transform(combined_transform2);

  dog = body1 + body2;

  Eigen::Matrix4f translation_matrix3 = get_translation(Eigen::Vector3f(0, -0.5, 0.5)); 
  Eigen::Matrix4f rotation_matrix3 = get_rotation(0, Eigen::Vector3f(0, 0, 1));    
  Eigen::Matrix4f scaling_matrix3 = get_scaling(Eigen::Vector3f(0.2, 1.5, 0.2));  
  Eigen::Matrix4f combined_transform3 = translation_matrix3 * rotation_matrix3 * scaling_matrix3;

  Mesh leg1 = cube.transform(combined_transform3);

  dog = leg1 + dog;

  Eigen::Matrix4f translation_matrix4 = get_translation(Eigen::Vector3f(0, -0.5, -0.5)); 
  Eigen::Matrix4f rotation_matrix4 = get_rotation(0, Eigen::Vector3f(0, 0, 1));    
  Eigen::Matrix4f scaling_matrix4 = get_scaling(Eigen::Vector3f(0.2, 1.5, 0.2));  
  Eigen::Matrix4f combined_transform4 = translation_matrix4 * rotation_matrix4 * scaling_matrix4;

  Mesh leg2 = cube.transform(combined_transform4);

  dog = leg2 + dog;

  Eigen::Matrix4f translation_matrix5 = get_translation(Eigen::Vector3f(-1.8, -0.3, 0.3)); 
  Eigen::Matrix4f rotation_matrix5 = get_rotation(-15, Eigen::Vector3f(0, 0, 1));    
  Eigen::Matrix4f scaling_matrix5 = get_scaling(Eigen::Vector3f(0.3, 0.5, 0.2));  
  Eigen::Matrix4f combined_transform5 = translation_matrix5 * rotation_matrix5 * scaling_matrix5;

  Mesh leg3 = cube.transform(combined_transform5);

  dog = leg3 + dog;

  translation_matrix1 = get_translation(Eigen::Vector3f(-1.8, -0.3, -0.3)); 
  rotation_matrix1 = get_rotation(-15, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.3, 0.5, 0.2));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh leg4 = cube.transform(combined_transform1);

  dog = leg4 + dog;

  translation_matrix1 = get_translation(Eigen::Vector3f(-2.0, -0.5, 0.3)); 
  rotation_matrix1 = get_rotation(-45, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.3, 0.5, 0.2));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh leg5 = cube.transform(combined_transform1);

  translation_matrix1 = get_translation(Eigen::Vector3f(-2.0, -0.5, -0.3)); 
  rotation_matrix1 = get_rotation(-45, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.3, 0.5, 0.2));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh leg6 = cube.transform(combined_transform1);

  dog = dog + leg5 + leg6;

  translation_matrix1 = get_translation(Eigen::Vector3f(-2.1, -0.9, 0.3)); 
  rotation_matrix1 = get_rotation(0, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.2, 0.6, 0.2));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh leg7 = cube.transform(combined_transform1);

  translation_matrix1 = get_translation(Eigen::Vector3f(-2.1, -0.9, -0.3)); 
  rotation_matrix1 = get_rotation(0, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.2, 0.6, 0.2));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh leg8 = cube.transform(combined_transform1);

  dog = dog + leg7 + leg8;

  translation_matrix1 = get_translation(Eigen::Vector3f(-2.0, 0.1, 0)); 
  rotation_matrix1 = get_rotation(-30, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.2, 0.4, 0.2));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh tail = cube.transform(combined_transform1);

  dog = dog + tail;

  translation_matrix1 = get_translation(Eigen::Vector3f(-2.0, -1.2, 0.3)); 
  rotation_matrix1 = get_rotation(0, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.2, 0.15, 0.2));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh feet1 = cube.transform(combined_transform1);

  translation_matrix1 = get_translation(Eigen::Vector3f(-2.0, -1.2, -0.3)); 
  rotation_matrix1 = get_rotation(0, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.2, 0.15, 0.2));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh feet2 = cube.transform(combined_transform1);

  dog = dog + feet1 + feet2;

  translation_matrix1 = get_translation(Eigen::Vector3f(0.1, -1.175, 0.5)); 
  rotation_matrix1 = get_rotation(0, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.2, 0.15, 0.2));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh feet3 = cube.transform(combined_transform1);

  translation_matrix1 = get_translation(Eigen::Vector3f(0.1, -1.175, -0.5)); 
  rotation_matrix1 = get_rotation(0, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.2, 0.15, 0.2));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh feet4 = cube.transform(combined_transform1);

  dog = dog + feet3 + feet4;

  translation_matrix1 = get_translation(Eigen::Vector3f(0.5, 0.5, 0)); 
  rotation_matrix1 = get_rotation(-20, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.5, 1, 0.5));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh neck = cube.transform(combined_transform1);

  dog = dog + neck;

  translation_matrix1 = get_translation(Eigen::Vector3f(0.7, 1.05, 0)); 
  rotation_matrix1 = get_rotation(0, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.5, 0.3, 0.5));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh head1 = cube.transform(combined_transform1);

  dog = dog + head1;

  translation_matrix1 = get_translation(Eigen::Vector3f(0.6, 0.9, 0)); 
  rotation_matrix1 = get_rotation(0, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.3, 0.4, 0.8));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh ear = cube.transform(combined_transform1);

  dog = dog + ear;

  translation_matrix1 = get_translation(Eigen::Vector3f(0.95, 0.9, 0)); 
  rotation_matrix1 = get_rotation(-20, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.15, 0.3, 0.45));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh head2 = cube.transform(combined_transform1);

  dog = dog + head2;

  translation_matrix1 = get_translation(Eigen::Vector3f(1.1, 0.85, 0)); 
  rotation_matrix1 = get_rotation(-20, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.25, 0.4, 0.7));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh head3 = cube.transform(combined_transform1);

  dog = dog + head3;

  translation_matrix1 = get_translation(Eigen::Vector3f(1.27, 0.95, 0)); 
  rotation_matrix1 = get_rotation(-45, Eigen::Vector3f(0, 0, 1));    
  scaling_matrix1 = get_scaling(Eigen::Vector3f(0.12, 0.12, 0.12));  
  combined_transform1 = translation_matrix1 * rotation_matrix1 * scaling_matrix1;

  Mesh nose = cube.transform(combined_transform1);

  dog = dog + nose;

  
  // Transform the cube mesh to build the dog.


  // You can use Mesh::transform to transform the cube mesh.
  // Mesh new_mesh = cube.transform(transform_matrix);
  // You can use "+" operator to add meshes together.
  // dog_mesh = cube + new_mesh;
  return dog;
}

int main(int argc, const char **argv) {
  float angle = 30;
  Eigen::Vector3f axis = Eigen::Vector3f(0, 1, 0);
  Eigen::Vector3f translation = Eigen::Vector3f(0, 0, 0);
  bool command_line = false;
  std::string filename = "output.png";

  if (argc >= 3) {
    command_line = true;
    angle = std::stof(argv[2]);  // -r by default
    if (argc == 4) {
      filename = std::string(argv[3]);
    } else 
      return 0;
  }

  // load mesh primitives
  Mesh cube_mesh;
  cube_mesh.load_obj("../model/cube.obj");
  auto dog_mesh = create_dog_mesh(cube_mesh);
  dog_mesh.save_obj("dog.obj");

  rst::rasterizer r(700, 700);

  Eigen::Vector3f eye_pos = {0, 2, 15};

  auto pos_id = r.load_positions(dog_mesh.vertices);
  auto ind_id = r.load_indices(dog_mesh.faces);

  int key = 0;
  int frame_count = 0;

  if (command_line) {
    r.clear(rst::Buffers::Color | rst::Buffers::Depth);

    r.set_model(get_model_matrix(angle, axis, translation));
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

    r.draw(pos_id, ind_id, rst::Primitive::Triangle);
    cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
    image.convertTo(image, CV_8UC3, 1.0f);

    cv::imwrite(filename, image);

    return 0;
  }

  while (key != 27) {
    r.clear(rst::Buffers::Color | rst::Buffers::Depth);

    r.set_model(get_model_matrix(angle, axis, translation));
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

    r.draw(pos_id, ind_id, rst::Primitive::Triangle);

    cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cv::imshow("image", image);
    key = cv::waitKey(10);

    std::cout << "frame count: " << frame_count++ << '\n';

    if (key == 'a') {
      eye_pos.x() -= 1;
    } else if (key == 'd') {
      eye_pos.x() += 1;
    } else if (key == 'w') {
      eye_pos.y() += 1;
    } else if (key == 's') {
      eye_pos.y() -= 1;
    } else if (key == 'q') {
      eye_pos.z() -= 1;
    } else if (key == 'e') {
      eye_pos.z() += 1;
    } else if (key == 'j') {
      angle += 10;
    } else if (key == 'k') {
      angle -= 10;
    }
    std::cout << "eye_pos: " << eye_pos.transpose() << std::endl;
  }

  return 0;
}
