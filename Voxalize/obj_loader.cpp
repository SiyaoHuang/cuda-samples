#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "obj_loader.h"

// Read .obj file and parse vertices and faces
void readObjFile(const char* filename, std::vector<Vertex>& vertices, std::vector<Face>& faces) {
    std::ifstream objFile(filename);
    std::cout << "getting file" << filename << std::endl;
    if (!objFile.is_open()) {
        std::cout << "does not open " << std::endl;
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    std::cout << "start reading" << std::endl;
    std::string line;
    while (std::getline(objFile, line)) {
        if (line.empty() || line[0] == '#') {
            continue; // Skip comments and empty lines
        }

        char type;
        std::istringstream iss(line);
        iss >> type;

        if (type == 'v') {
            Vertex vertex;
            iss >> vertex.x >> vertex.y >> vertex.z;
            vertices.push_back(vertex);
        }
        else if (type == 'f') {
            Face face;
            iss >> face.v1 >> face.v2 >> face.v3;
            // Adjust indices (assuming 1-based indices in .obj file)
            face.v1--; face.v2--; face.v3--;
            faces.push_back(face);
        }
    }
}

// Read .obj file and parse vertices and faces
void readObjFile(const char* filename, std::vector<float3>& vertices, std::vector<Face>& faces) {
    std::ifstream objFile(filename);
    std::cout << "getting file" << filename << std::endl;
    if (!objFile.is_open()) {
        std::cout << "does not open " << std::endl;
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    std::cout << "start reading" << std::endl;
    std::string line;
    while (std::getline(objFile, line)) {
        if (line.empty() || line[0] == '#') {
            continue; // Skip comments and empty lines
        }

        char type;
        std::istringstream iss(line);
        iss >> type;

        if (type == 'v') {
            float3 vertex;
            iss >> vertex.x >> vertex.y >> vertex.z;
            vertices.push_back(vertex);
        }
        else if (type == 'f') {
            Face face;
            iss >> face.v1 >> face.v2 >> face.v3;
            // Adjust indices (assuming 1-based indices in .obj file)
            face.v1--; face.v2--; face.v3--;
            faces.push_back(face);
        }
    }
}