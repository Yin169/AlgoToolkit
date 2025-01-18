#ifndef SU2MESHADAPTER_HPP
#define SU2MESHADAPTER_HPP

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include "SpectralElementMethod.hpp"

class SU2MeshAdapter {
public:
    SU2MeshAdapter(const std::string& mesh_file) : mesh_file_(mesh_file) {}

    void readMesh(SpectralElementMethod<double>& sem) {
        std::ifstream file(mesh_file_);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open mesh file: " + mesh_file_);
        }

        std::string line;
        std::vector<std::vector<double>> nodes;
        std::vector<std::vector<size_t>> elements;
        std::map<std::string, std::vector<std::vector<size_t>>> markers;

        size_t num_dimensions = 0;
        size_t num_points = 0;
        size_t num_elements_mesh = 0;
        size_t num_markers_mesh = 0;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;

            if (keyword == "NDIME=") {
                iss >> num_dimensions;
                if (num_dimensions != sem.num_dimensions) {
                    throw std::runtime_error("Mesh dimensions do not match SEM dimensions.");
                }
            } else if (keyword == "NPOIN=") {
                iss >> num_points;
                nodes.resize(num_points);
                for (size_t i = 0; i < num_points; ++i) {
                    std::getline(file, line);
                    std::istringstream point_stream(line);
                    std::vector<double> coords(num_dimensions);
                    for (size_t d = 0; d < num_dimensions; ++d) {
                        if (!(point_stream >> coords[d])) {
                            throw std::runtime_error("Invalid coordinates for node " + std::to_string(i));
                        }
                    }
                    nodes[i] = coords;
                }
            } else if (keyword == "NELEM=") {
                iss >> num_elements_mesh;
                elements.resize(num_elements_mesh);
                for (size_t i = 0; i < num_elements_mesh; ++i) {
                    std::getline(file, line);
                    std::istringstream elem_stream(line);
                    size_t elem_type, num_nodes;
                    if (!(elem_stream >> elem_type >> num_nodes)) {
                        throw std::runtime_error("Invalid element data at element " + std::to_string(i));
                    }
                    std::vector<size_t> node_indices(num_nodes);
                    for (size_t j = 0; j < num_nodes; ++j) {
                        if (!(elem_stream >> node_indices[j])) {
                            throw std::runtime_error("Invalid node index at element " + std::to_string(i) + ", node " + std::to_string(j));
                        }
                        node_indices[j]--; // Convert to zero-based indexing
                    }
                    elements[i] = node_indices;
                }
            } else if (keyword == "NMARK=") {
                iss >> num_markers_mesh;
                for (size_t i = 0; i < num_markers_mesh; ++i) {
                    std::getline(file, line);
                    std::istringstream marker_stream(line);
                    std::string marker_tag;
                    if (!(marker_stream >> marker_tag)) {
                        throw std::runtime_error("Invalid marker tag at marker " + std::to_string(i));
                    }
                    size_t num_elements_marker = 0;
                    std::getline(file, line);
                    std::istringstream marker_elem_stream(line);
                    if (!(marker_elem_stream >> num_elements_marker)) {
                        throw std::runtime_error("Invalid number of elements in marker " + marker_tag);
                    }
                    std::vector<std::vector<size_t>> boundary_elements(num_elements_marker);
                    for (size_t j = 0; j < num_elements_marker; ++j) {
                        std::getline(file, line);
                        std::istringstream boundary_elem_stream(line);
                        size_t elem_type, num_nodes;
                        if (!(boundary_elem_stream >> elem_type >> num_nodes)) {
                            throw std::runtime_error("Invalid boundary element data at marker " + marker_tag + ", element " + std::to_string(j));
                        }
                        std::vector<size_t> node_indices(num_nodes);
                        for (size_t k = 0; k < num_nodes; ++k) {
                            if (!(boundary_elem_stream >> node_indices[k])) {
                                throw std::runtime_error("Invalid node index at marker " + marker_tag + ", element " + std::to_string(j) + ", node " + std::to_string(k));
                            }
                            node_indices[k]--; // Convert to zero-based indexing
                        }
                        boundary_elements[j] = node_indices;
                    }
                    markers[marker_tag] = boundary_elements;
                }
            }
        }

        // Build face_to_elements map
        std::map<std::vector<size_t>, std::vector<size_t>> face_to_elements;
        for (size_t elem_idx = 0; elem_idx < elements.size(); ++elem_idx) {
            const auto& elem_nodes = elements[elem_idx];
            auto elem_faces = get_faces(elem_nodes, num_dimensions);
            for (const auto& face : elem_faces) {
                face_to_elements[face].push_back(elem_idx);
            }
        }

        // Assign neighboring elements and identify boundary faces
        for (auto& face_pair : face_to_elements) {
            const auto& face_nodes = face_pair.first;
            const auto& elem_list = face_pair.second;
            if (elem_list.size() == 1) {
                // Boundary face
                size_t elem_idx = elem_list[0];
                sem.setBoundaryCondition(elem_idx, face_nodes, markers);
            } else if (elem_list.size() == 2) {
                // Internal face, assign neighboring elements
                size_t elem_idx1 = elem_list[0];
                size_t elem_idx2 = elem_list[1];
                sem.setNeighboringElements(elem_idx1, elem_idx2, face_nodes);
            } else {
                throw std::runtime_error("Face shared by more than two elements, which should not happen.");
            }
        }

        // Initialize SEM with the mesh data
        initializeSEM(sem, nodes, elements, face_to_elements, markers);
    }

private:
    std::string mesh_file_;

    std::vector<std::vector<size_t>> get_faces(const std::vector<size_t>& nodes, size_t num_dimensions) {
        std::vector<std::vector<size_t>> faces;
        if (num_dimensions == 2) {
            if (nodes.size() == 3) { // Triangle
                faces.push_back({nodes[0], nodes[1]});
                faces.push_back({nodes[1], nodes[2]});
                faces.push_back({nodes[2], nodes[0]});
            } else if (nodes.size() == 4) { // Quadrilateral
                faces.push_back({nodes[0], nodes[1]});
                faces.push_back({nodes[1], nodes[2]});
                faces.push_back({nodes[2], nodes[3]});
                faces.push_back({nodes[3], nodes[0]});
            }
        } else if (num_dimensions == 3) {
            if (nodes.size() == 4) { // Tetrahedron
                faces.push_back({nodes[0], nodes[1], nodes[2]});
                faces.push_back({nodes[1], nodes[2], nodes[3]});
                faces.push_back({nodes[0], nodes[2], nodes[3]});
                faces.push_back({nodes[0], nodes[1], nodes[3]});
            } else if (nodes.size() == 8) { // Hexahedron
                faces.push_back({nodes[0], nodes[1], nodes[2], nodes[3]});
                faces.push_back({nodes[4], nodes[5], nodes[6], nodes[7]});
                faces.push_back({nodes[0], nodes[1], nodes[5], nodes[4]});
                faces.push_back({nodes[2], nodes[3], nodes[7], nodes[6]});
                faces.push_back({nodes[0], nodes[3], nodes[7], nodes[4]});
                faces.push_back({nodes[1], nodes[2], nodes[6], nodes[5]});
            }
        }
        for (auto& face : faces) {
            std::sort(face.begin(), face.end());
        }
        return faces;
    }

    void initializeSEM(SpectralElementMethod<double>& sem,
                       const std::vector<std::vector<double>>& nodes,
                       const std::vector<std::vector<size_t>>& elements,
                       const std::map<std::vector<size_t>, std::vector<size_t>>& face_to_elements,
                       const std::map<std::string, std::vector<std::vector<size_t>>>& markers) {
        // Populate SEM's element connectivity
        sem.element_connectivity.resize(elements.size());
        for (size_t elem_idx = 0; elem_idx < elements.size(); ++elem_idx) {
            sem.element_connectivity[elem_idx].global_indices = elements[elem_idx];
            sem.element_connectivity[elem_idx].face_nodes = get_faces(elements[elem_idx], sem.num_dimensions);
            sem.element_connectivity[elem_idx].neighboring_elements.resize(sem.element_connectivity[elem_idx].face_nodes.size(), -1);
        }
        // Further initialization as required by SEM
    }
};

#endif // SU2MESHADAPTER_HPP