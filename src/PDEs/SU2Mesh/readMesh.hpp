#ifndef READ_MESH_HPP
#define READ_MESH_HPP

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include "../../Obj/VectorObj.hpp"
#include "../FDM/NavierStoke.hpp"

class SU2Mesh {
private:
    int dimension;
    int num_elements;
    int num_points;
    
    std::vector<std::vector<int>> elements;
    std::vector<std::vector<double>> points;
    std::vector<int> element_types;
    
    // Boundary markers
    std::unordered_map<std::string, std::vector<std::pair<int, int>>> boundaries;
    
    // For structured grid conversion
    int nx, ny, nz;
    double dx, dy, dz;
    bool is_structured;
    
    // Element type mapping (SU2 element types)
    // 9 = QUAD
    // 10 = TETRA
    // 12 = HEXA
    std::unordered_map<int, std::string> element_type_map = {
        {3, "LINE"},
        {5, "TRIANGLE"},
        {9, "QUAD"},
        {10, "TETRA"},
        {12, "HEXA"},
        {13, "PRISM"},
        {14, "PYRAMID"}
    };

public:
    SU2Mesh() : dimension(0), num_elements(0), num_points(0), nx(0), ny(0), nz(0), 
                dx(0.0), dy(0.0), dz(0.0), is_structured(false) {}
    
    bool readMeshFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open mesh file " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '%') continue;
            
            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;
            
            if (keyword == "NDIME=") {
                iss >> dimension;
            } 
            else if (keyword == "NELEM=") {
                iss >> num_elements;
                elements.resize(num_elements);
                element_types.resize(num_elements);
                
                // Read element connectivity
                for (int i = 0; i < num_elements; i++) {
                    if (!std::getline(file, line)) {
                        std::cerr << "Error: Unexpected end of file while reading elements" << std::endl;
                        return false;
                    }
                    
                    std::istringstream elem_iss(line);
                    int elem_type;
                    elem_iss >> elem_type;
                    element_types[i] = elem_type;
                    
                    // Determine number of nodes for this element type
                    int num_nodes;
                    if (elem_type == 9) { // QUAD
                        num_nodes = 4;
                    } else if (elem_type == 10) { // TETRA
                        num_nodes = 4;
                    } else if (elem_type == 12) { // HEXA
                        num_nodes = 8;
                    } else {
                        std::cerr << "Warning: Unknown element type " << elem_type << std::endl;
                        num_nodes = 0;
                        while (elem_iss >> elem_type) num_nodes++;
                        continue;
                    }
                    
                    elements[i].resize(num_nodes);
                    for (int j = 0; j < num_nodes; j++) {
                        elem_iss >> elements[i][j];
                    }
                }
            } 
            else if (keyword == "NPOIN=") {
                iss >> num_points;
                points.resize(num_points, std::vector<double>(dimension));
                
                // Read point coordinates
                for (int i = 0; i < num_points; i++) {
                    if (!std::getline(file, line)) {
                        std::cerr << "Error: Unexpected end of file while reading points" << std::endl;
                        return false;
                    }
                    
                    std::istringstream point_iss(line);
                    for (int j = 0; j < dimension; j++) {
                        point_iss >> points[i][j];
                    }
                    
                    // Skip point index if present
                    int point_idx;
                    if (point_iss >> point_idx) {
                        // Optional: verify point index matches expected value
                        if (point_idx != i) {
                            std::cerr << "Warning: Point index mismatch. Expected " << i 
                                      << ", got " << point_idx << std::endl;
                        }
                    }
                }
            }
            else if (keyword == "NMARK=") {
                int num_markers;
                iss >> num_markers;
                
                for (int i = 0; i < num_markers; i++) {
                    // Read marker tag
                    if (!std::getline(file, line)) break;
                    std::istringstream marker_iss(line);
                    std::string marker_keyword, marker_tag;
                    marker_iss >> marker_keyword >> marker_tag;
                    
                    if (marker_keyword != "MARKER_TAG=") {
                        std::cerr << "Error: Expected MARKER_TAG, got " << marker_keyword << std::endl;
                        continue;
                    }
                    
                    // Read number of elements for this marker
                    if (!std::getline(file, line)) break;
                    std::istringstream marker_elems_iss(line);
                    std::string marker_elems_keyword;
                    int num_marker_elems;
                    marker_elems_iss >> marker_elems_keyword >> num_marker_elems;
                    
                    if (marker_elems_keyword != "MARKER_ELEMS=") {
                        std::cerr << "Error: Expected MARKER_ELEMS, got " << marker_elems_keyword << std::endl;
                        continue;
                    }
                    
                    // Read marker elements
                    std::vector<std::pair<int, int>> marker_elements;
                    for (int j = 0; j < num_marker_elems; j++) {
                        if (!std::getline(file, line)) break;
                        std::istringstream marker_elem_iss(line);
                        int elem_type, node1, node2;
                        marker_elem_iss >> elem_type >> node1 >> node2;
                        
                        // Store boundary element (assuming line elements for boundaries)
                        marker_elements.push_back(std::make_pair(node1, node2));
                    }
                    
                    boundaries[marker_tag] = marker_elements;
                }
            }
        }
        
        file.close();
        
        // Check if the mesh is structured (regular grid)
        detectStructuredGrid();
        
        return true;
    }
    
    void detectStructuredGrid() {
        // For 2D meshes, try to detect if it's a structured grid
        if (dimension == 2) {
            // Find unique x and y coordinates
            std::vector<double> x_coords, y_coords;
            for (const auto& point : points) {
                if (std::find(x_coords.begin(), x_coords.end(), point[0]) == x_coords.end()) {
                    x_coords.push_back(point[0]);
                }
                if (std::find(y_coords.begin(), y_coords.end(), point[1]) == y_coords.end()) {
                    y_coords.push_back(point[1]);
                }
            }
            
            // Sort coordinates
            std::sort(x_coords.begin(), x_coords.end());
            std::sort(y_coords.begin(), y_coords.end());
            
            nx = x_coords.size();
            ny = y_coords.size();
            
            // Check if number of points matches nx * ny
            if (num_points == nx * ny) {
                is_structured = true;
                
                // Calculate grid spacing (assuming uniform grid)
                if (nx > 1) dx = (x_coords.back() - x_coords.front()) / (nx - 1);
                if (ny > 1) dy = (y_coords.back() - y_coords.front()) / (ny - 1);
                
                std::cout << "Detected structured grid: " << nx << " x " << ny << std::endl;
                std::cout << "Grid spacing: dx = " << dx << ", dy = " << dy << std::endl;
            }
        }
        else if (dimension == 3) {
            // Similar logic for 3D meshes
            std::vector<double> x_coords, y_coords, z_coords;
            for (const auto& point : points) {
                if (std::find(x_coords.begin(), x_coords.end(), point[0]) == x_coords.end()) {
                    x_coords.push_back(point[0]);
                }
                if (std::find(y_coords.begin(), y_coords.end(), point[1]) == y_coords.end()) {
                    y_coords.push_back(point[1]);
                }
                if (std::find(z_coords.begin(), z_coords.end(), point[2]) == z_coords.end()) {
                    z_coords.push_back(point[2]);
                }
            }
            
            std::sort(x_coords.begin(), x_coords.end());
            std::sort(y_coords.begin(), y_coords.end());
            std::sort(z_coords.begin(), z_coords.end());
            
            nx = x_coords.size();
            ny = y_coords.size();
            nz = z_coords.size();
            
            if (num_points == nx * ny * nz) {
                is_structured = true;
                
                if (nx > 1) dx = (x_coords.back() - x_coords.front()) / (nx - 1);
                if (ny > 1) dy = (y_coords.back() - y_coords.front()) / (ny - 1);
                if (nz > 1) dz = (z_coords.back() - z_coords.front()) / (nz - 1);
                
                std::cout << "Detected structured grid: " << nx << " x " << ny << " x " << nz << std::endl;
                std::cout << "Grid spacing: dx = " << dx << ", dy = " << dy << ", dz = " << dz << std::endl;
            }
        }
    }
    
    // Getters
    int getDimension() const { return dimension; }
    int getNumElements() const { return num_elements; }
    int getNumPoints() const { return num_points; }
    bool isStructured() const { return is_structured; }
    
    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }
    double getDx() const { return dx; }
    double getDy() const { return dy; }
    double getDz() const { return dz; }
    
    const std::vector<std::vector<int>>& getElements() const { return elements; }
    const std::vector<std::vector<double>>& getPoints() const { return points; }
    const std::vector<int>& getElementTypes() const { return element_types; }
    
    // Get point coordinates
    std::vector<double> getPoint(int index) const {
        if (index >= 0 && index < num_points) {
            return points[index];
        }
        return std::vector<double>(dimension, 0.0);
    }
    
    // Get element connectivity
    std::vector<int> getElement(int index) const {
        if (index >= 0 && index < num_elements) {
            return elements[index];
        }
        return std::vector<int>();
    }
    
    // Get element type
    int getElementType(int index) const {
        if (index >= 0 && index < num_elements) {
            return element_types[index];
        }
        return -1;
    }
    
    // Get boundary elements for a specific marker
    const std::vector<std::pair<int, int>>& getBoundaryElements(const std::string& marker) const {
        static const std::vector<std::pair<int, int>> empty;
        auto it = boundaries.find(marker);
        if (it != boundaries.end()) {
            return it->second;
        }
        return empty;
    }
    
    // Get all boundary markers
    std::vector<std::string> getBoundaryMarkers() const {
        std::vector<std::string> markers;
        for (const auto& boundary : boundaries) {
            markers.push_back(boundary.first);
        }
        return markers;
    }
    
    // For structured grids, get point index from i,j,k indices
    int getPointIndex(int i, int j, int k = 0) const {
        if (!is_structured) return -1;
        if (i < 0 || i >= nx || j < 0 || j >= ny || (dimension == 3 && (k < 0 || k >= nz))) {
            return -1;
        }
        
        if (dimension == 2) {
            return i + j * nx;
        } else {
            return i + j * nx + k * nx * ny;
        }
    }
    
    // Get i,j,k indices from point index (for structured grids)
    void getIJK(int point_index, int& i, int& j, int& k) const {
        if (!is_structured || point_index < 0 || point_index >= num_points) {
            i = j = k = -1;
            return;
        }
        
        if (dimension == 2) {
            i = point_index % nx;
            j = point_index / nx;
            k = 0;
        } else {
            i = point_index % nx;
            j = (point_index / nx) % ny;
            k = point_index / (nx * ny);
        }
    }
    
    // Convert to VectorObj for use with NavierStokes solver
    void getVelocityGrid(VectorObj<double>& u, VectorObj<double>& v, VectorObj<double>& w) const {
        if (!is_structured) {
            std::cerr << "Error: Cannot create velocity grid for unstructured mesh" << std::endl;
            return;
        }
        
        if (dimension == 2) {
            u.resize(nx * ny, 0.0);
            v.resize(nx * ny, 0.0);
            w.resize(nx * ny, 0.0);
        } else {
            u.resize(nx * ny * nz, 0.0);
            v.resize(nx * ny * nz, 0.0);
            w.resize(nx * ny * nz, 0.0);
        }
    }
    
    // Get coordinates for a structured grid point
    void getCoordinates(int i, int j, int k, double& x, double& y, double& z) const {
        if (!is_structured) {
            x = y = z = 0.0;
            return;
        }
        
        if (i >= 0 && i < nx && j >= 0 && j < ny) {
            int idx = getPointIndex(i, j, k);
            if (idx >= 0 && idx < num_points) {
                x = points[idx][0];
                y = points[idx][1];
                z = (dimension == 3 && points[idx].size() > 2) ? points[idx][2] : 0.0;
                return;
            }
        }
        
        x = y = z = 0.0;
    }
    
    // Setup for NavierStokes solver
    template <typename TNum>
    void setupNavierStokesSolver(NavierStokesSolver3D<TNum>& solver) const {
        if (!is_structured) {
            std::cerr << "Error: NavierStokes solver requires structured grid" << std::endl;
            return;
        }
        
        // Set initial conditions to zero
        solver.setInitialConditions(
            [](TNum x, TNum y, TNum z) { return 0.0; },  // u
            [](TNum x, TNum y, TNum z) { return 0.0; },  // v
            [](TNum x, TNum y, TNum z) { return 0.0; }   // w
        );
        
        // Set default boundary conditions (can be overridden by user)
        solver.setBoundaryConditions(
            [](TNum x, TNum y, TNum z, TNum t) { return 0.0; },  // u
            [](TNum x, TNum y, TNum z, TNum t) { return 0.0; },  // v
            [](TNum x, TNum y, TNum z, TNum t) { return 0.0; }   // w
        );
    }
    
    // Apply boundary conditions from SU2 mesh to NavierStokes solver
    template <typename TNum>
    void applyBoundaryConditions(NavierStokesSolver3D<TNum>& solver, 
                                const std::unordered_map<std::string, 
                                std::function<void(TNum, TNum, TNum, TNum, TNum&, TNum&, TNum&)>>& bc_funcs) const {
        if (!is_structured) {
            std::cerr << "Error: Cannot apply boundary conditions for unstructured mesh" << std::endl;
            return;
        }
        
        // Set boundary conditions based on the provided functions for each marker
        solver.setBoundaryConditions(
            [this, &bc_funcs](TNum x, TNum y, TNum z, TNum t) {
                TNum u = 0.0, v = 0.0, w = 0.0;
                
                // Find which boundary this point belongs to
                for (const auto& marker_pair : bc_funcs) {
                    const std::string& marker = marker_pair.first;
                    const auto& bc_func = marker_pair.second;
                    
                    // Check if point is on this boundary
                    // This is a simplified check - in practice, you'd need more sophisticated logic
                    bool on_boundary = false;
                    
                    // For x = 0 boundary
                    if (marker == "left" && std::abs(x) < 1e-10) on_boundary = true;
                    // For x = max boundary
                    else if (marker == "right" && std::abs(x - (nx-1)*dx) < 1e-10) on_boundary = true;
                    // For y = 0 boundary
                    else if (marker == "bottom" && std::abs(y) < 1e-10) on_boundary = true;
                    // For y = max boundary
                    else if (marker == "top" && std::abs(y - (ny-1)*dy) < 1e-10) on_boundary = true;
                    
                    if (on_boundary) {
                        bc_func(x, y, z, t, u, v, w);
                        return u;  // Return u component
                    }
                }
                
                return TNum(0.0);
            },
            [this, &bc_funcs](TNum x, TNum y, TNum z, TNum t) {
                TNum u = 0.0, v = 0.0, w = 0.0;
                
                // Similar logic as above
                for (const auto& marker_pair : bc_funcs) {
                    const std::string& marker = marker_pair.first;
                    const auto& bc_func = marker_pair.second;
                    
                    bool on_boundary = false;
                    if (marker == "left" && std::abs(x) < 1e-10) on_boundary = true;
                    else if (marker == "right" && std::abs(x - (nx-1)*dx) < 1e-10) on_boundary = true;
                    else if (marker == "bottom" && std::abs(y) < 1e-10) on_boundary = true;
                    else if (marker == "top" && std::abs(y - (ny-1)*dy) < 1e-10) on_boundary = true;
                    
                    if (on_boundary) {
                        bc_func(x, y, z, t, u, v, w);
                        return v;  // Return v component
                    }
                }
                
                return TNum(0.0);
            },
            [this, &bc_funcs](TNum x, TNum y, TNum z, TNum t) {
                TNum u = 0.0, v = 0.0, w = 0.0;
                
                // Similar logic as above
                for (const auto& marker_pair : bc_funcs) {
                    const std::string& marker = marker_pair.first;
                    const auto& bc_func = marker_pair.second;
                    
                    bool on_boundary = false;
                    if (marker == "left" && std::abs(x) < 1e-10) on_boundary = true;
                    else if (marker == "right" && std::abs(x - (nx-1)*dx) < 1e-10) on_boundary = true;
                    else if (marker == "bottom" && std::abs(y) < 1e-10) on_boundary = true;
                    else if (marker == "top" && std::abs(y - (ny-1)*dy) < 1e-10) on_boundary = true;
                    
                    if (on_boundary) {
                        bc_func(x, y, z, t, u, v, w);
                        return w;  // Return w component
                    }
                }
                
                return TNum(0.0);
            }
        );
    }
    
    // Export solution to VTK format for visualization
    template <typename TNum>
    bool exportToVTK(const std::string& filename, 
                    const VectorObj<TNum>& u, 
                    const VectorObj<TNum>& v, 
                    const VectorObj<TNum>& w, 
                    const VectorObj<TNum>& p) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
            return false;
        }
        
        // Write VTK header
        file << "# vtk DataFile Version 3.0\n";
        file << "SU2 Mesh with NavierStokes Solution\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_GRID\n";
        
        if (dimension == 2) {
            file << "DIMENSIONS " << nx << " " << ny << " 1\n";
            file << "POINTS " << nx * ny << " float\n";
            
            // Write point coordinates
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    double x, y, z;
                    getCoordinates(i, j, 0, x, y, z);
                    file << x << " " << y << " " << z << "\n";
                }
            }
        } else {
            file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
            file << "POINTS " << nx * ny * nz << " float\n";
            
            // Write point coordinates
            for (int k = 0; k < nz; k++) {
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        double x, y, z;
                        getCoordinates(i, j, k, x, y, z);
                        file << x << " " << y << " " << z << "\n";
                    }
                }
            }
        }
        
        // Write point data
        int num_points = (dimension == 2) ? nx * ny : nx * ny * nz;
        file << "POINT_DATA " << num_points << "\n";
        
        // Write velocity vector field
        file << "VECTORS velocity float\n";
        for (int idx = 0; idx < num_points; idx++) {
            file << u[idx] << " " << v[idx] << " " << w[idx] << "\n";
        }
        
        // Write pressure scalar field
        file << "SCALARS pressure float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int idx = 0; idx < num_points; idx++) {
            file << p[idx] << "\n";
        }
        
        file.close();
        return true;
    }
};

#endif // READ_MESH_HPP