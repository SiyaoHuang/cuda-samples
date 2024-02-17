
#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include <vector>
#include <string>

#include <stdio.h>
#include "obj_loader.h"
#include <iostream>

#include <GL/freeglut.h>

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

#pragma endregion morton

// 相机参数
float cameraRadius = 0.5f; // 相机到原点的距离
float cameraAngle = 0.0f;  // 相机绕原点旋转的角度

std::vector<Vertex> vertices; // Your parsed vertices
std::vector<Face> faces;     // Your parsed faces

void range() {
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

int main(int argc, char** argv)
{
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
    range();
    run(argc, argv);
    return 0;
}