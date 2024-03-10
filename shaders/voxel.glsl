
#version 430

layout(local_size_x = 32, local_size_y = 32) in;
layout(rgba32f, binding = 0) uniform image2D img_output; // location = 0
layout(rgba32f, binding = 1) uniform image2D accumulation_buffer; // location = 14, this is the previous frames
layout(rgba8, binding = 2) uniform image3D world_data; // location = 7
layout(location = 3) uniform vec2 resolution;
layout(location = 4) uniform float time;
layout(location = 5) uniform vec3 cam_pos;
layout(location = 6) uniform vec3 cam_dir;
layout(location = 7) uniform float cam_fov;
layout(location = 8) uniform vec3 world_size;
layout(location = 9) uniform vec3 world_sky_a_color; // Used for sky gradient.
layout(location = 10) uniform vec3 world_sky_b_color;
layout(location = 11) uniform vec3 world_sky_angle;
layout(location = 12) uniform float sun_size;
layout(location = 13) uniform vec3 sun_color;
layout(location = 14) uniform bool show_normals;
layout(location = 15) uniform int frame_count;
layout(location = 16) uniform float sun_intensity;
layout(location = 17) uniform float roughness;
layout(location = 18) uniform float metallic;
layout(location = 19) uniform int max_bounces;

struct Camera {
    vec3 origin;
    vec3 direction;

    float focal_length;
    float viewport_height;
    float viewport_width;
    float fov;

    int bounces;
    int samples_per_pixel;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct HitPayload {
    vec3 normal;
    vec3 color;
    float hitDistance;
    bool hit;
    bool hitsun;
    vec3 hitpos;
};

// Gold Noise function
float PHI = 1.61803398874989484820459 * 00000.1; // Golden Ratio   
float PI = 3.14159265358979323846264 * 00000.1; // PI
float SRT = 1.41421356237309504880169 * 10000.0; // Square Root of Two

float random_0t1(in vec2 coordinate, in float seed) {
    return fract(sin(dot(coordinate * seed, vec2(PHI, PI))) * SRT);
}

float rand(vec2 co) {
    // Use time as a seed.
    float seed = time;
    return random_0t1(co, seed);
}

vec3 random_in_unit_sphere(vec2 pixel_coords) {
    float phi = 2.0 * 3.14159265359 * rand(pixel_coords);
    float costheta = 2.0 * rand(pixel_coords + vec2(37.0, 17.0)) - 1.0;
    float sintheta = sqrt(1.0 - costheta * costheta);
    return vec3(sintheta * cos(phi), sintheta * sin(phi), costheta);
}

HitPayload Trace(vec3 p0, vec3 d) {
    // Raytrace the world.
    vec3 normal = vec3(0.0, 0.0, 0.0);
    bool hit = false;
    vec3 hitpos = vec3(0.0, 0.0, 0.0);
    bool hitsun = false;
    vec3 color = vec3(0.0, 0.0, 0.0);

    float t = 0.0;
    float t_max = 1000.0;

    float t_step = 0.01;

    for(float i = 0; i < t_max; i += t_step) {
        if(t > t_max) {
            break;
        }

        vec3 p = p0 + t * d;

        if(p.x < 0.0 || p.x > world_size.x ||
            p.y < 0.0 || p.y > world_size.y ||
            p.z < 0.0 || p.z > world_size.z) {
            // Hit the world boundary.
            break;
        }

        vec4 voxel = imageLoad(world_data, ivec3(p));
        if(voxel.a > 0.0) {
            // Hit a voxel.
            normal = vec3(0.0, 0.0, 0.0);
            float epsilon = 0.05;
            vec3 dx = vec3(epsilon, 0.0, 0.0);
            vec3 dy = vec3(0.0, epsilon, 0.0);
            vec3 dz = vec3(0.0, 0.0, epsilon);

            float nx, ny, nz;
            // Check if it's a boundary voxel.
            if(imageLoad(world_data, ivec3(p + dx)).a == 0.0) {
                nx = -1.0;
            } else if(imageLoad(world_data, ivec3(p - dx)).a == 0.0) {
                nx = 1.0;
            } else {
                nx = 0.0;
            }

            if(imageLoad(world_data, ivec3(p + dy)).a == 0.0) {
                ny = -1.0;
            } else if(imageLoad(world_data, ivec3(p - dy)).a == 0.0) {
                ny = 1.0;
            } else {
                ny = 0.0;
            }

            if(imageLoad(world_data, ivec3(p + dz)).a == 0.0) {
                nz = -1.0;
            } else if(imageLoad(world_data, ivec3(p - dz)).a == 0.0) {
                nz = 1.0;
            } else {
                nz = 0.0;
            }
            normal = normalize(vec3(nx, ny, nz));

            hit = true;
            hitpos = p;

            color = vec3(voxel.r, voxel.g, voxel.b);

            HitPayload payload;
            payload.normal = normal;
            payload.color = color;
            payload.hitDistance = t;
            payload.hit = hit;
            payload.hitsun = hitsun;
            payload.hitpos = hitpos;

            return payload;
        }

        t += t_step;
    }

    // Sky gradient, using the angle of the ray and the world_sky_angle.
    float angle = dot(normalize(d), world_sky_angle);
    color = mix(world_sky_a_color, world_sky_b_color, 0.5 * (1.0 + angle));
    hit = false;

    // If the ray is pointing at the sun, change the color to be the sun color.
    if(-angle > 1.40 - sun_size) {
        color = sun_color;
        hitsun = true;
        hit = true;
    }

    HitPayload payload;
    payload.normal = normal;
    payload.color = color;
    payload.hitDistance = t;
    payload.hit = hit;
    payload.hitsun = hitsun;
    payload.hitpos = hitpos;

    return payload;
}

vec3 Raytrace(Ray ray, Camera camera, vec2 pixel_coords) {
    vec3 color = vec3(0.0);
    vec3 attenuation = vec3(0.6);

    vec3 contribution = vec3(1.0);
    vec3 light = vec3(0.0);

    int bounces = camera.bounces;

    // The sun will function as the light source,
    // the trace function will return the sun color and if it hit the sun.
    for(int bounce = 0; bounce < bounces; bounce++) {
        HitPayload payload = Trace(ray.origin, ray.direction);

        if(!payload.hit) {
            // If the ray hit the atmosphere or sun, stop tracing.
            float angle = dot(normalize(ray.direction), world_sky_angle);
            vec3 skyColor = mix(world_sky_a_color, world_sky_b_color, 0.5 * (1.0 + angle));

            light += contribution * skyColor;
            break;
        }

        if(payload.hitsun) {
            // If the ray hit the sun, stop tracing.
            vec3 sunColor = payload.color;
            light += contribution * sunColor * sun_intensity;
            break;
        }

        vec3 hitpos = payload.hitpos;
        contribution *= payload.color;

        vec3 reflected = reflect(ray.direction, payload.normal);
        reflected = normalize(reflected);

        // Add some randomness to the reflected ray.
        vec3 randomOffset = random_in_unit_sphere(pixel_coords);
        reflected += roughness * randomOffset;
        reflected = normalize(reflected);

        ray.origin = hitpos + 0.1 * reflected;
        ray.direction = reflected;
    }

    color = light;

    return color;
}

void main() {
    vec4 pixel = vec4(0.0, 0.0, 0.0, 1.0); // Clear color.

    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    // Aspect ratio.
    float aspect_ratio = resolution.x / resolution.y;
    int image_width = int(resolution.x);
    int image_height = int(resolution.y);

    // Camera
    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;

    Camera camera;
    camera.origin = cam_pos;
    camera.direction = cam_dir;
    camera.fov = cam_fov;
    camera.focal_length = 1.0;
    camera.viewport_height = viewport_height;
    camera.viewport_width = viewport_width;
    camera.samples_per_pixel = 10;
    camera.bounces = int(max_bounces);

    // Ray
    float u = float(pixel_coords.x) / float(image_width - 1);
    float v = float(pixel_coords.y) / float(image_height - 1);

    vec3 color = vec3(0.0, 0.0, 0.0);

    vec3 camera_right = normalize(cross(camera.direction, vec3(0.0, 1.0, 0.0)));
    vec3 camera_up = normalize(cross(camera_right, camera.direction));

    float fov = camera.fov;
    float theta = fov * 3.14159265358979323846 / 180.0;
    float half_height = tan(theta / 2.0);
    float half_width = aspect_ratio * half_height;
    vec3 ray_dir = camera.direction;
    ray_dir += (2.0 * u - 1.0) * half_width * camera_right;
    ray_dir += (2.0 * v - 1.0) * half_height * camera_up;
    Ray ray = Ray(cam_pos, ray_dir);

    if(show_normals) {
        HitPayload payload = Trace(ray.origin, ray.direction);
        vec3 normal = payload.normal;
        bool hit = payload.hit;

        if(hit) {
            // If a normal is negative, make it positive.
            normal = abs(normal);
            color = normal;
        } else {
            color = payload.color;
        }
    } else {
        color = Raytrace(ray, camera, pixel_coords);
    }

    pixel = vec4(color, 1.0);

    if(frame_count > 0) {
        // Accumulate the color.
        vec4 prev_color = imageLoad(accumulation_buffer, pixel_coords);
        pixel = (prev_color * float(frame_count) + pixel) / float(frame_count + 1);
    }

    // Write the pixel to the image.
    imageStore(img_output, pixel_coords, pixel);
}
