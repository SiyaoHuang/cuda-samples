
#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include <vector>
#include <string>

#include <stdio.h>
#include "obj_loader.h"
#include <iostream>

#include <GL/freeglut.h>

struct PointData {
    float x;
    float y;
    float z;
};

__device__ unsigned int globalIndex = 0; // 全局索引，初始值为 0

__global__ void AddDataToBuffer(PointData* buffer, int numPoints) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 假设你有一些点数据，这里只是示例
    PointData newPoint;
    newPoint.x = 1.0f; // 设置 x 坐标
    newPoint.y = 2.0f; // 设置 y 坐标
    newPoint.z = 3.0f; // 设置 z 坐标

    // 在不同线程中添加数据到缓冲区
    if (tid < numPoints) {
        unsigned int index = atomicAdd(&globalIndex, 1); // 递增全局索引
        buffer[index] = newPoint;
    }
}


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
// 相机参数
float cameraRadius = 0.5f; // 相机到原点的距离
float cameraAngle = 0.0f;  // 相机绕原点旋转的角度

std::vector<Vertex> vertices; // Your parsed vertices
std::vector<Face> faces;     // Your parsed faces

void  range() {
    // 初始化边界框的最小和最大值
    float min_x = vertices[0].x;
    float min_y = vertices[0].y;
    float min_z = vertices[0].z;
    float max_x = vertices[0].x;
    float max_y = vertices[0].y;
    float max_z = vertices[0].z;

    // 遍历所有顶点，更新最小和最大值
    for (const auto& vertex : vertices) {
        min_x = std::min(min_x, vertex.x);
        min_y = std::min(min_y, vertex.y);
        min_z = std::min(min_z, vertex.z);
        max_x = std::max(max_x, vertex.x);
        max_y = std::max(max_y, vertex.y);
        max_z = std::max(max_z, vertex.z);
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

#if 1 // triangle
    glBegin(GL_TRIANGLES);
    for (const auto& face : faces) {
        glColor3f(1.0, 0.0, 0.0);
        glVertex3f(vertices[face.v1].x, vertices[face.v1].y, vertices[face.v1].z);
        glColor3f(0.0, 1.0, 0.0);
        glVertex3f(vertices[face.v2].x, vertices[face.v2].y, vertices[face.v2].z);
        glColor3f(0.0, 0.0, 1.0);
        glVertex3f(vertices[face.v3].x, vertices[face.v3].y, vertices[face.v3].z);
    }
#else// point
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    for (const auto& vertice : vertices) {
        glColor3f(1.0f, 1.0f, 1.0f); // White color for particles
        glVertex2f(vertice.x, vertice.y);
    }
#endif
    glEnd();

    // 检查 OpenGL 错误状态
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error: " << gluErrorString(error) << std::endl;
    }
    else {
        std::cout << "sucess" << std::endl;
    }

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

void addPoints() {
    int numPoints = 256; // 假设你有 256 个点
    PointData* devBuffer;
    cudaMalloc(&devBuffer, numPoints * sizeof(PointData));

    // 调用内核函数
    AddDataToBuffer << <1, numPoints >> > (devBuffer, numPoints-20);

    // 将结果传回主机（如果需要的话）

    // 获取最终的全局索引
    unsigned int finalIndex;
    cudaMemcpyFromSymbol(&finalIndex, globalIndex, sizeof(unsigned int));

    std::cout << "Total points added: " << finalIndex + 1 << std::endl;

    // 释放内存
    cudaFree(devBuffer);
}

int main(int argc, char** argv)
{
    

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    addPoints();

    const char* objFilename = "D:\\Download\\bunny.obj";

    readObjFile(objFilename, vertices, faces);
    std::cout << "vertices:" << vertices.size() << ", faces:" << faces.size() << std::endl;
    range();
    run(argc, argv);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
