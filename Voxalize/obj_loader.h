#pragma once
#include <vector>
#include <vector_types.h>

struct Vertex {
    float x, y, z; // Position (X, Y, Z)
};

struct Face {
    int v1, v2, v3; // Vertex indices for a triangle face
};

// Read .obj file and parse vertices and faces
void readObjFile(const char* filename, std::vector<Vertex>& vertices, std::vector<Face>& faces);

void readObjFile(const char* filename, std::vector<float3>& vertices, std::vector<Face>& faces);