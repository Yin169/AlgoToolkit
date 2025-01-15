#ifndef SU2_MESH_ADAPTER_HPP
#define SU2_MESH_ADAPTER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <unordered_map>

class SU2Mesh {
public:
    // Node structure
    struct Node {
        size_t index;
        std::vector<double> coordinates;
    };

    // Element structure
    struct Element {
        size_t index;
        std::string type;
        std::vector<size_t> connectivity;
    };

    // Boundary marker structure
    struct BoundaryMarker {
        std::string name;
        std::vector<Element> elements;
    };

    // Mesh data
    size_t dimensionality;
    std::vector<Node> nodes;
    std::vector<Element> elements;
    std::unordered_map<std::string, BoundaryMarker> boundary_markers;

    // Parse SU2 mesh file
    void loadMesh(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open SU2 mesh file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;

            if (keyword == "NDIME=") {
                iss >> dimensionality;
            } else if (keyword == "NPOIN=") {
                size_t num_nodes;
                iss >> num_nodes;
                nodes.reserve(num_nodes);
                for (size_t i = 0; i < num_nodes; ++i) {
                    Node node;
                    node.index = i;
                    node.coordinates.resize(dimensionality);
                    file >> node.index;
                    for (size_t d = 0; d < dimensionality; ++d) {
                        file >> node.coordinates[d];
                    }
                    nodes.push_back(node);
                }
            } else if (keyword == "NELEM=") {
                size_t num_elements;
                iss >> num_elements;
                elements.reserve(num_elements);
                for (size_t i = 0; i < num_elements; ++i) {
                    Element element;
                    size_t num_nodes_in_element;
                    iss >> element.index >> element.type >> num_nodes_in_element;
                    element.connectivity.resize(num_nodes_in_element);
                    for (size_t j = 0; j < num_nodes_in_element; ++j) {
                        file >> element.connectivity[j];
                    }
                    elements.push_back(element);
                }
            } else if (keyword == "NMARK=") {
                size_t num_markers;
                iss >> num_markers;
                for (size_t i = 0; i < num_markers; ++i) {
                    std::getline(file, line);
                    std::istringstream marker_stream(line);
                    std::string marker_keyword;
                    marker_stream >> marker_keyword;

                    if (marker_keyword == "MARKER_TAG=") {
                        BoundaryMarker marker;
                        marker_stream >> marker.name;

                        std::getline(file, line);
                        std::istringstream marker_elems_stream(line);
                        marker_elems_stream >> marker_keyword; // Skip "MARKER_ELEMS="

                        size_t num_marker_elements;
                        marker_elems_stream >> num_marker_elements;
                        marker.elements.reserve(num_marker_elements);

                        for (size_t j = 0; j < num_marker_elements; ++j) {
                            Element element;
                            size_t num_nodes_in_element;
                            file >> element.index >> element.type >> num_nodes_in_element;
                            element.connectivity.resize(num_nodes_in_element);
                            for (size_t k = 0; k < num_nodes_in_element; ++k) {
                                file >> element.connectivity[k];
                            }
                            marker.elements.push_back(element);
                        }
                        boundary_markers[marker.name] = marker;
                    }
                }
            }
        }
    }

    // Debug: Print mesh summary
    void printSummary() const {
        std::cout << "Dimensionality: " << dimensionality << "\n";
        std::cout << "Number of Nodes: " << nodes.size() << "\n";
        std::cout << "Number of Elements: " << elements.size() << "\n";
        std::cout << "Boundary Markers: " << boundary_markers.size() << "\n";
    }
};

#endif // SU2_MESH_ADAPTER_HPP
