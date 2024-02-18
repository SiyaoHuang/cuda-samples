
#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include <vector>
#include <string>

#include <stdio.h>
#include "obj_loader.h"
#include "helper_math.h"
#include <iostream>

#include <GL/freeglut.h>

#define DIVID_COUNT 50;
#define TEST_CODE 0;


// 相机参数
float cameraRadius = 0.5f; // 相机到原点的距离
float cameraAngle = 0.0f;  // 相机绕原点旋转的角度

//std::vector<Vertex> vertices; // Your parsed vertices
std::vector<float3> vertices; // Your parsed vertices
std::vector<int3> faces;     // Your parsed faces
std::vector<float3> particles;

float3 hostminVal;
float3 hostmaxVal;

#pragma region morton
#define __all__ __host__ __device__
__all__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__all__ unsigned int morton3D(float x, float y, float z)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

__all__ unsigned int invertBits(unsigned int v) {
    v = v & 0x49249249u;
    v = (v | (v >> 2)) & 0xC30C30C3u;
    v = (v | (v >> 4)) & 0x0F00F00Fu;
    v = (v | (v >> 8)) & 0xFF0000FFu;
    v = (v | (v >> 16)) & 0x0000FFFFu;
    return v;
}

// Converts a 30-bit Morton code back to 3D float coordinates
__all__ void mortonToFloat(unsigned int morton, float& x, float& y, float& z) {
    unsigned int xx = invertBits(morton >> 0);
    unsigned int yy = invertBits(morton >> 1);
    unsigned int zz = invertBits(morton >> 2);

    // Normalize the values back to the [0,1] range
    x = static_cast<float>(xx) / 1024.0f;
    y = static_cast<float>(yy) / 1024.0f;
    z = static_cast<float>(zz) / 1024.0f;
}
#if TEST_CODE
void test_morton_convert() {
    float x = 0.1;
    float y = 0.9;
    float z = 0.34;
    std::cout << "test morton" << std::endl;
    std::cout << x << "," << y << "," << z << std::endl;
    uint code = morton3D(x, y, z);
    std::cout << "morton:" << code << std::endl;
    float rx = 0;
    float ry = 0;
    float rz = 0;
    mortonToFloat(code, rx, ry, rz);
    std::cout << "re morton" << rx << "," << ry << "," << rz << std::endl;
}
#endif // TEST_CODE


#pragma endregion

#pragma region boudingbox cuda
__device__ float atomicMinFloat(float* addr, float value) {
    int* address_as_i = (int*)addr;
    int old = *address_as_i;
    while (value < __int_as_float(old)) {
        old = atomicCAS(address_as_i, old, __float_as_int(value));
    }
    return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* addr, float value) {
    int* address_as_i = (int*)addr;
    int old = *address_as_i;
    while (value > __int_as_float(old)) {
        old = atomicCAS(address_as_i, old, __float_as_int(value));
    }
    return __int_as_float(old);
}

__global__ void boundingboxKernel(const float3* data, int size, float3 * minVal, float3* maxVal)
{
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if (i >= size) {
        return;
    }
    
    float3 point = data[i];
    atomicMinFloat(&minVal->x, point.x);
    atomicMinFloat(&minVal->y, point.y);
    atomicMinFloat(&minVal->z, point.z);

    atomicMaxFloat(&maxVal->x, point.x);
    atomicMaxFloat(&maxVal->y, point.y);
    atomicMaxFloat(&maxVal->z, point.z);
}

__global__ void boundingboxKernelV2(const float3* data, int size, float3* minVal, float3* maxVal)
{
    extern __shared__ float3 sharedData[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if (i >= size) {
        return;
    }

    sharedData[tid] = data[i];

    __syncthreads();

    float3 localMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 localMax = make_float3(FLT_MIN, FLT_MIN, FLT_MIN);

    if (tid == 0)
    {    
        for (int i = 0; i < blockDim.x && blockIdx.x * blockDim.x + i < size; i++) {
            localMin = fminf(localMin, sharedData[i]);

            localMax = fmaxf(localMax, sharedData[i]);
        }
    }

    if (tid == 0) {
        atomicMinFloat(&minVal->x, localMin.x);
        atomicMinFloat(&minVal->y, localMin.y);
        atomicMinFloat(&minVal->z, localMin.z);

        atomicMaxFloat(&maxVal->x, localMax.x);
        atomicMaxFloat(&maxVal->y, localMax.y);
        atomicMaxFloat(&maxVal->z, localMax.z);
    }

}


void cudaBoudingBox(std::vector<float3>& input)
{

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    float3* data;
    hostminVal = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    hostmaxVal = make_float3(FLT_MIN, FLT_MIN, FLT_MIN);
    int size = input.size();
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    float3* deviceMinVal;
    float3* deviceMaxVal;

    cudaMalloc(&data, size * sizeof(float3));
    cudaMalloc(&deviceMinVal, sizeof(float3));
    cudaMalloc(&deviceMaxVal, sizeof(float3));

    cudaMemcpy(deviceMinVal, &hostminVal, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaxVal, &hostmaxVal, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(data, input.data(), size * sizeof(float3), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
#if 1
    boundingboxKernel << < gridSize, blockSize >> > (data, size, deviceMinVal, deviceMaxVal);
#else
    const size_t smSz = blockSize * sizeof(float3);
    boundingboxKernelV2 << < gridSize, blockSize, smSz >> > (data, size, deviceMinVal, deviceMaxVal);
#endif
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(&hostminVal, deviceMinVal, sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostmaxVal, deviceMaxVal, sizeof(float3), cudaMemcpyDeviceToHost);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Bounding Box Min: (" << hostminVal.x << ", " << hostminVal.y << ", " << hostminVal.z << ")\n";
    std::cout << "Bounding Box Max: (" << hostmaxVal.x << ", " << hostmaxVal.y << ", " << hostmaxVal.z << ")\n";

    cudaFree(data);
    cudaFree(deviceMinVal);
    cudaFree(deviceMaxVal);
}

#pragma endregion

#pragma region boudingbox cpu
void cpuBoundingBox() {
    // 初始化边界框的最小和最大值
    float min_x = vertices[0].x;
    float min_y = vertices[0].y;
    float min_z = vertices[0].z;
    float max_x = vertices[0].x;
    float max_y = vertices[0].y;
    float max_z = vertices[0].z;

    // 遍历所有顶点，更新最小和最大值
    for (const auto& vertex : vertices) {
        min_x = std::fmin(min_x, vertex.x);
        min_y = std::fmin(min_y, vertex.y);
        min_z = std::fmin(min_z, vertex.z);
        max_x = std::fmax(max_x, vertex.x);
        max_y = std::fmax(max_y, vertex.y);
        max_z = std::fmax(max_z, vertex.z);
    }

    // 打印边界框信息
    std::cout << "Bounding Box:" << std::endl;
    std::cout << "Min X: " << min_x << std::endl;
    std::cout << "Max X: " << max_x << std::endl;
    std::cout << "Min Y: " << min_y << std::endl;
    std::cout << "Max Y: " << max_y << std::endl;
    std::cout << "Min Z: " << min_z << std::endl;
    std::cout << "Max Z: " << max_z << std::endl;
}
#pragma endregion

#pragma region all particles within bouding box
#define GENRATE_BY_CUDA 1
__device__ __host__ float divid_length(float3 min, float3 maxIn) {
    float3 range = maxIn - min;
    float maxv = max(range.x, range.y);
    maxv = max(maxv, range.z);
    return maxv / DIVID_COUNT;
}

__global__ void cuda_generate_particle_within_bouding_box(float3* particlebuffer, float3 min_p, float divid_len, int x_len, int y_len, int z_len) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int index = i;
    int count = x_len * y_len * z_len;
    if (i >= count) {
        return;
    }
    int zid = i / (x_len * y_len);
    i = i - zid * (x_len * y_len);
    int yid = i / (x_len);
    i = i - yid * (x_len);
    int xid = i;

    particlebuffer[index] = min_p + make_float3(divid_len * xid, divid_len * yid, divid_len * zid);
}

void generate_particle_within_bouding_box(float3 min_p, float3 max_p) {
    float divid_len = divid_length(min_p, max_p);

    std::cout << "divid len:" << divid_len << std::endl;
    if (divid_len == 0) {
        return;
    }
#if GENRATE_BY_CUDA
    float3* particlebuffer;
    float3 dif_p = max_p - min_p;
    int x_len = dif_p.x / divid_len+1;
    int y_len = dif_p.y / divid_len+1;
    int z_len = dif_p.z / divid_len+1;
    int count = x_len * y_len * z_len;
    std::cout << "len:" << x_len << ", " << y_len << ", " << z_len << std::endl;
    cudaMalloc(&particlebuffer, count * sizeof(float3));
    int blocksize = 256;
    int gridcount = (count + blocksize - 1) / blocksize;
    cuda_generate_particle_within_bouding_box << <gridcount, blocksize  >> > (particlebuffer, min_p, divid_len, x_len, y_len, z_len);

    particles.resize(count);
    cudaMemcpy(particles.data(), particlebuffer, count * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaFree(particlebuffer);

#else
    for (float x = min_p.x; x <= max_p.x; x += divid_len) {
        for (float y = min_p.y; y <= max_p.y; y += divid_len) {
            for (float z = min_p.z; z <= max_p.z; z += divid_len) {
                particles.push_back(make_float3(x, y, z));
            }
        }
    }
#endif
}
#pragma endregion

#pragma region paricle from triangles

#define MAX_PARTICLES_COUNT 70000
__device__ __inline__ void put_data(float3* data, int* current_index, float3 value) {
    int returnIndex = atomicAdd(current_index, 1);
    if (returnIndex >= MAX_PARTICLES_COUNT) {
        return;
    }
    data[returnIndex] = value;

}

__device__ __inline__ int3 toIndex(float3 min, float3 val, float divid_len) {
    float3 dif = val - min;
    return make_int3(
        (int)(dif.x / divid_len),
        (int)(dif.y / divid_len),
        (int)(dif.z / divid_len)
    );
}

__device__ __inline__ float3 toPosition(float3 min, int x, int y, int z, float divid_len) {
    return make_float3(
        min.x + x * divid_len,
        min.y + y * divid_len,
        min.z + z * divid_len
    );
}

__device__ __inline__ float3 toPosition(float3 min, int3 val, float divid_len) {
    return make_float3(
        min.x + val.x * divid_len,
        min.y + val.y * divid_len,
        min.z + val.z * divid_len
    );
}

__device__ __inline__ float3 toGrid(float3 min, float3 val, float divid_len) {
    float3 dif = val - min;
    return make_float3(
        min.x + (uint)(dif.x / divid_len) * divid_len,
        min.y + (uint)(dif.y / divid_len) * divid_len,
        min.z + (uint)(dif.z / divid_len) * divid_len
    );
}

__global__ void cuda_generate_particles_from_triangles(float3* vertices, int3* faces, float3* particles, size_t face_count, float3 min, float divid_len, int* particle_count) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index >= face_count) {
        return;
    }
    int3 face = faces[index];
    float3 v1 = vertices[face.x];
    float3 v2 = vertices[face.y];
    float3 v3 = vertices[face.z];
#if 0 // vertice grid position
    put_data(particles, particle_count, toGrid(min, v1, divid_len));
    put_data(particles, particle_count, toGrid(min, v2, divid_len));
    put_data(particles, particle_count, toGrid(min, v3, divid_len));
#endif

    float3 minv = v1;
    minv = fminf(minv, v2);
    minv = fminf(minv, v3);
    float3 maxv = v1;
    maxv = fmaxf(maxv, v2);
    maxv = fmaxf(maxv, v3);
    int3 lowIndex = toIndex(min, minv, divid_len);
    int3 highIndex = toIndex(min, maxv, divid_len);

    for (int x = lowIndex.x; x <= highIndex.x; x++) {
        for (int y = lowIndex.y; y <= highIndex.y; y++) {
            for (int z = lowIndex.z; z <= highIndex.z; z++) {
                put_data(particles, particle_count, toPosition(min, x, y, z, divid_len));
            }
        }
    }

}

void generate_particles_from_triangles() {
    float divid_len = divid_length(hostminVal, hostmaxVal);
    
    float3* deviceVertices;
    int3* deviceFaces;
    float3* deviceParticles;
    int* particle_count;

    cudaMalloc(&deviceVertices, vertices.size() * sizeof(float3));
    cudaMalloc(&deviceFaces, faces.size() * sizeof(int3));
    cudaMalloc(&deviceParticles, MAX_PARTICLES_COUNT * sizeof(float3));
    cudaMalloc(&particle_count, sizeof(int));

    cudaMemcpy(deviceVertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFaces, faces.data(), faces.size() * sizeof(int3), cudaMemcpyHostToDevice);
    int host_particle_count = 0;
    cudaMemcpy(&particle_count, &host_particle_count, sizeof(int), cudaMemcpyHostToDevice);

    size_t face_count = faces.size();
    size_t blocksize = 256;
    size_t gridcount = (face_count + blocksize - 1) / blocksize;

    cuda_generate_particles_from_triangles << <gridcount, blocksize >> > (deviceVertices, deviceFaces, deviceParticles, face_count, hostminVal, divid_len, particle_count);
    cudaMemcpy(&host_particle_count, particle_count, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "getting particle count:" << host_particle_count << std::endl;
    host_particle_count = min(host_particle_count, MAX_PARTICLES_COUNT);

    particles.resize(host_particle_count);
    cudaMemcpy(particles.data(), deviceParticles, host_particle_count * sizeof(float3), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceVertices);
    cudaFree(deviceFaces);
    cudaFree(deviceParticles);
    //cudaFree(deviceCount);
}

#pragma endregion

#pragma region freeglut render and interact
void reshape(int width, int height) {
    // 防止除以零
    if (height == 0) {
        height = 1;
    }

    // 设置视口
    glViewport(0, 0, width, height);

    // 设置投影矩阵
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // 设置视角
    gluPerspective(45.0, (float)width / (float)height, 0.1, 100.0);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // 设置相机位置
    float cameraX = cameraRadius * std::cos(cameraAngle);
    float cameraY = cameraRadius * std::sin(cameraAngle);
    gluLookAt(cameraX, 0.5, cameraY,  // 相机位置
        0.0, 0.0, 0.0,          // 观察点
        0.0, 1.0, 0.0);         // 上方向
#if 0
    // triangle
    glBegin(GL_TRIANGLES);
    for (const auto& face : faces) {
        glColor3f(1.0, 0.0, 0.0);
        glVertex3f(vertices[face.x].x, vertices[face.x].y, vertices[face.x].z);
        glColor3f(0.0, 1.0, 0.0);
        glVertex3f(vertices[face.y].x, vertices[face.y].y, vertices[face.y].z);
        glColor3f(0.0, 0.0, 1.0);
        glVertex3f(vertices[face.z].x, vertices[face.z].y, vertices[face.z].z);
    }
    glEnd();
#endif

    // point
    //glPointSize(3.0f);
    glBegin(GL_POINTS);
    for (const auto& vertice : particles) {
        glColor3f(1.0f, 1.0f, 1.0f); // White color for particles
        glVertex3f(vertice.x, vertice.y, vertice.z);
    }
    glEnd();

#if TEST_CODE
    // 检查 OpenGL 错误状态
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error: " << gluErrorString(error) << std::endl;
    }
    else {
        std::cout << "sucess" << std::endl;
    }
#endif

    glutSwapBuffers();
}
void specialKeys(int key, int x, int y) {
    const float rotationSpeed = 0.01f;
    if (key == GLUT_KEY_LEFT) {
        cameraAngle -= rotationSpeed;
    }
    else if (key == GLUT_KEY_RIGHT) {
        cameraAngle += rotationSpeed;
    }
    glutPostRedisplay();
}

void run(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutCreateWindow("Polygon Viewer");
    glEnable(GL_DEPTH_TEST);
    // 设置重塑回调
    glutReshapeFunc(reshape);
    // Set display callback
    glutDisplayFunc(display);
    glutSpecialFunc(specialKeys);

    // Your other initialization code

    // Enter GLUT main loop
    glutMainLoop();
}
#pragma endregion

int main(int argc, char** argv)
{
#if TEST_CODE
    test_morton_convert();
#endif
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    const char* objFilename = "D:\\Download\\bunny.obj";

    readObjFile(objFilename, vertices, faces);
    std::cout << "vertices:" << vertices.size() << ", faces:" << faces.size() << std::endl;
    cudaBoudingBox(vertices);
#if 0
    generate_particle_within_bouding_box(hostminVal, hostmaxVal);
#else
    generate_particles_from_triangles();
#endif

    cpuBoundingBox();
    run(argc, argv);
    return 0;
}