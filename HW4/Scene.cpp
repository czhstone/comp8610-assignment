//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}


void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            pos.happened=true;  // area light that has emission exists
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}


// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    Vector3f hitColor = Vector3f(0);
    auto inter = intersect(ray);
    if (!inter.happened)return backgroundColor;

    Vector3f hitPoint = inter.coords;
    Vector3f N = inter.normal; // normal
    Vector2f st = inter.tcoords; // texture coordinates
    Vector3f dir = ray.direction;

    if (inter.material->m_type == EMIT) {
        return inter.material->m_emission;
    }else if (inter.material->m_type == DIFFUSE || TASK_N<3) {
        Vector3f lightAmt = 0, specularColor = 0, diffuseColor =0;

        // sample area light
        int light_sample=4;
        for (int i = 0; i < light_sample && TASK_N >= 5; ++i) {
            Intersection lightInter;
            float pdf_light = 0.0f;
            sampleLight(lightInter, pdf_light);  // sample a point on the area light
            // TODO: task 5 soft shadow
            
            // Calculate the direction vector from the hit point to the sampled light point
            Vector3f lightDir = normalize(lightInter.coords - hitPoint);
            // Calculate the squared distance between the hit point and the sampled light point
            float distanceSquared = dotProduct(lightInter.coords - hitPoint, lightInter.coords - hitPoint);

            // Create a shadow ray starting from the hit point towards the light point, with a slight offset to avoid self-intersection
            Ray shadowRay(hitPoint + N * EPSILON, lightDir);
            // Check for intersections along the shadow ray path
            Intersection shadowInter = intersect(shadowRay);

            // Determine if the sampled light point is in shadow
            bool inShadow = shadowInter.happened && (shadowInter.tnear - (lightInter.coords - shadowInter.coords).norm() < 0);
            if (!inShadow) {
                // Calculate the dot product between the normal and light direction
                float lightDotNormal = std::max(dotProduct(N, lightDir), 0.0f);
                // Accumulate the contribution from the area light to the final hit color
                hitColor += inter.obj->evalDiffuseColor(st) * lightInter.material->getEmission() *
                            lightInter.material->eval(lightDir, N) * lightDotNormal / distanceSquared / pdf_light;
            }

        }
        // Average the accumulated color across all light samples
        hitColor = hitColor / float(light_sample);

        // TODO: task 1.3 Basic shading

        for (const auto& light : this->lights) {
            // Calculate direction towards the light source
            Vector3f lightDirection = normalize(light->position - hitPoint);

            // Calculate squared distance between the hit point and the light source
            float lightDistanceSquared = dotProduct(light->position - hitPoint, light->position - hitPoint);

            // Create a new ray slightly offset to avoid self-shadowing
            Ray Ray(hitPoint + N * EPSILON, lightDirection);

            // Check for objects between the hit point and the light source (shadow)
            Intersection shadowInter = intersect(Ray);
            bool inShadow = shadowInter.happened && (shadowInter.tnear * shadowInter.tnear < lightDistanceSquared);

            if (!inShadow) {
                // Compute the Lambertian diffuse component
                float lightDotNormal = std::max(dotProduct(lightDirection, N), 0.0f);
                diffuseColor = inter.obj->evalDiffuseColor(st) * inter.material->Kd * lightDotNormal * light->intensity;

                // Calculate reflection direction for the specular component
                Vector3f reflectDir = reflect(-lightDirection, N);
                float specFactor = std::max(dotProduct(dir, -reflectDir), 0.0f);
                float specularFactor = pow(specFactor, inter.material->specularExponent);

                // Calculate the specular color contribution
                specularColor = specularFactor * light->intensity * inter.material->Ks;
            }
        }
        // Sum up the diffuse and specular contributions
        hitColor += diffuseColor + specularColor;

    

    } else if (inter.material->m_type == GLASS && TASK_N>=3) {
        // TODO: task 3 glass material

        if (depth > 7) { // Limit recursion depth to avoid infinite loops
            return Vector3f(0.0);
        }

        // Initialize colors for reflection and refraction
        Vector3f reflectedColor = Vector3f(0.0f);
        Vector3f refractedColor = Vector3f(0.0f);

        // Compute reflection direction
        Vector3f reflectionDirection = reflect(dir, N).normalized();
        // Determine the offset for reflection ray origin to avoid self-intersection
        Vector3f reflectionRayOffset = dotProduct(reflectionDirection, N) > 0 ? N * EPSILON : -N * EPSILON;
        // Create a new reflection ray
        Ray reflectionRay(hitPoint + reflectionRayOffset, reflectionDirection);
        // Trace the reflection ray and get the resulting color
        reflectedColor = castRay(reflectionRay, depth + 1);

        // Compute refraction direction using Snell's Law
        Vector3f refractionDirection = refract(dir, N, inter.material->ior).normalized();
        // Determine the offset for refraction ray origin to avoid self-intersection
        Vector3f refractionRayOffset = dotProduct(refractionDirection, N) > 0 ? N * EPSILON : -N * EPSILON;
        // Create a new refraction ray
        Ray refractionRay(hitPoint + refractionRayOffset, refractionDirection);
        // Trace the refraction ray and get the resulting color
        refractedColor = castRay(refractionRay, depth + 1);

        // Compute the Fresnel effect to determine the reflection coefficient
        float reflectionCoefficient = fresnel(dir, N, inter.material->ior);

        // Combine reflection and refraction colors based on the reflection coefficient
        hitColor = reflectedColor * reflectionCoefficient + refractedColor * (1.0f - reflectionCoefficient);

    }

    return hitColor;
}
