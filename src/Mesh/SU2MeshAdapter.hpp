#ifndef SU2MESHADAPTER_HPP
#define SU2MESHADAPTER_HPP

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
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
        std::vector<std::pair<std::string, std::vector<size_t>>> boundaries;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;

            if (keyword == "NDIME=") {
                size_t num_dimensions;
                iss >> num_dimensions;
                // Assuming the SEM class is already initialized with the correct number of dimensions
            } else if (keyword == "NPOIN=") {
                size_t num_points;
                iss >> num_points;
                nodes.resize(num_points);
                for (size_t i = 0; i < num_points; ++i) {
                    std::getline(file, line);
                    std::istringstream point_stream(line);
                    std::vector<double> coords(sem.num_dimensions);
                    for (size_t d = 0; d < sem.num_dimensions; ++d) {
                        point_stream >> coords[d];
                    }
                    nodes[i] = coords;
                }
            } else if (keyword == "NELEM=") {
                size_t num_elements;
                iss >> num_elements;
                elements.resize(num_elements);
                for (size_t i = 0; i < num_elements; ++i) {
                    std::getline(file, line);
                    std::istringstream elem_stream(line);
                    size_t elem_type, num_nodes;
                    elem_stream >> elem_type >> num_nodes;
                    std::vector<size_t> node_indices(num_nodes);
                    for (size_t j = 0; j < num_nodes; ++j) {
                        elem_stream >> node_indices[j];
                    }
                    elements[i] = node_indices;
                }
            } else if (keyword == "NMARK=") {
                size_t num_markers;
                iss >> num_markers;
                for (size_t i = 0; i < num_markers; ++i) {
                    std::getline(file, line);
                    std::istringstream marker_stream(line);
                    std::string marker_tag;
                    marker_stream >> marker_tag;
                    size_t num_elements_marker;
                    std::getline(file, line);
                    std::istringstream marker_elem_stream(line);
                    marker_elem_stream >> num_elements_marker;
                    std::vector<size_t> boundary_elements(num_elements_marker);
                    for (size_t j = 0; j < num_elements_marker; ++j) {
                        std::getline(file, line);
                        std::istringstream boundary_elem_stream(line);
                        size_t elem_type, num_nodes;
                        boundary_elem_stream >> elem_type >> num_nodes;
                        std::vector<size_t> node_indices(num_nodes);
                        for (size_t k = 0; k < num_nodes; ++k) {
                            boundary_elem_stream >> node_indices[k];
                        }
                        boundary_elements[j] = node_indices[0]; // Assuming boundary elements are points
                    }
                    boundaries.emplace_back(marker_tag, boundary_elements);
                }
            }
        }

        // Initialize the SEM class with the mesh data
        initializeSEM(sem, nodes, elements, boundaries);
    }

private:
    std::string mesh_file_;

    void initializeSEM(SpectralElementMethod<double>& sem, 
                       const std::vector<std::vector<double>>& nodes,
                       const std::vector<std::vector<size_t>>& elements,
                       const std::vector<std::pair<std::string, std::vector<size_t>>>& boundaries) {
        // Assuming the SEM class is already initialized with the correct number of elements and polynomial order
        // We need to set up the element connectivity and boundary conditions

        // Set up element connectivity
        for (size_t e = 0; e < elements.size(); ++e) {
            auto& conn = sem.element_connectivity[e];
            conn.global_indices = elements[e];
            // Initialize face nodes and neighboring elements (this is a simplified example)
            initializeElementFaces(sem, e, elements);
        }

        // Set up boundary conditions
        for (const auto& boundary : boundaries) {
            std::string marker_tag = boundary.first;
            const auto& boundary_nodes = boundary.second;
            for (size_t node_index : boundary_nodes) {
                std::vector<double> node_coords = nodes[node_index];
                // Apply boundary condition based on marker tag
                // This is a simplified example, assuming Dirichlet boundary conditions
                sem.boundary_condition = [](const std::vector<double>& coords) {
                    return 0.0; // Example: fixed value boundary condition
                };
            }
        }
    }

    void initializeElementFaces(SpectralElementMethod<double>& sem, size_t element_index, 
                                const std::vector<std::vector<size_t>>& elements) {
        auto& conn = sem.element_connectivity[element_index];
        conn.face_nodes.resize(2 * sem.num_dimensions);
        conn.neighboring_elements.resize(2 * sem.num_dimensions, -1);

        // Simplified example: assuming elements are ordered and neighbors are adjacent
        if (element_index > 0) {
            conn.neighboring_elements[0] = element_index - 1;
        }
        if (element_index < elements.size() - 1) {
            conn.neighboring_elements[1] = element_index + 1;
        }
    }
};

#endif // SU2MESHADAPTER_HPP