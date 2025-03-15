#ifndef MESH_OBJ_HPP
#define MESH_OBJ_HPP

#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <algorithm>
#include <cmath>
#include <array>
#include <tuple>
#include "../../Obj/VectorObj.hpp"

// Forward declaration
template <typename TNum> class MeshCell;
template <typename TNum> class MeshNode;

// Enum for refinement criteria
enum class RefinementCriterion {
    GRADIENT,
    VORTICITY,
    CUSTOM
};

// Enum for cell types
enum class CellType {
    FLUID,
    SOLID,
    BOUNDARY,
    GHOST
};

// Structure to represent a 3D point
template <typename TNum>
struct Point3D {
    TNum x, y, z;
    
    Point3D() : x(0), y(0), z(0) {}
    Point3D(TNum x_, TNum y_, TNum z_) : x(x_), y(y_), z(z_) {}
    
    TNum distance(const Point3D<TNum>& other) const {
        TNum dx = x - other.x;
        TNum dy = y - other.y;
        TNum dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

// Structure to represent a 3D vector
template <typename TNum>
struct Vector3D {
    TNum x, y, z;
    
    Vector3D() : x(0), y(0), z(0) {}
    Vector3D(TNum x_, TNum y_, TNum z_) : x(x_), y(y_), z(z_) {}
    
    TNum magnitude() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    Vector3D<TNum> normalize() const {
        TNum mag = magnitude();
        if (mag > 1e-10) {
            return Vector3D<TNum>(x/mag, y/mag, z/mag);
        }
        return *this;
    }
    
    Vector3D<TNum> operator+(const Vector3D<TNum>& other) const {
        return Vector3D<TNum>(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3D<TNum> operator-(const Vector3D<TNum>& other) const {
        return Vector3D<TNum>(x - other.x, y - other.y, z - other.z);
    }
    
    Vector3D<TNum> operator*(TNum scalar) const {
        return Vector3D<TNum>(x * scalar, y * scalar, z * scalar);
    }
    
    TNum dot(const Vector3D<TNum>& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    Vector3D<TNum> cross(const Vector3D<TNum>& other) const {
        return Vector3D<TNum>(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
};

// Node class for mesh
template <typename TNum>
class MeshNode {
public:
    int id;
    Point3D<TNum> position;
    std::vector<int> connectedCells;
    
    MeshNode() : id(-1) {}
    MeshNode(int id_, TNum x, TNum y, TNum z) : id(id_), position(x, y, z) {}
    
    void addCell(int cellId) {
        if (std::find(connectedCells.begin(), connectedCells.end(), cellId) == connectedCells.end()) {
            connectedCells.push_back(cellId);
        }
    }
};

// Cell class for mesh
template <typename TNum>
class MeshCell {
public:
    int id;
    std::vector<int> nodeIds;
    std::vector<int> neighborCells;
    CellType type;
    int level;  // Refinement level
    Point3D<TNum> center;
    TNum volume;
    
    // Fields stored at cell center
    TNum u, v, w, p;
    
    MeshCell() : id(-1), type(CellType::FLUID), level(0), volume(0) {}
    
    MeshCell(int id_, const std::vector<int>& nodes, int level_ = 0) 
        : id(id_), nodeIds(nodes), type(CellType::FLUID), level(level_), volume(0) {}
    
    void addNeighbor(int cellId) {
        if (std::find(neighborCells.begin(), neighborCells.end(), cellId) == neighborCells.end()) {
            neighborCells.push_back(cellId);
        }
    }
    
    void calculateCenter(const std::vector<MeshNode<TNum>>& nodes) {
        TNum x_sum = 0, y_sum = 0, z_sum = 0;
        for (int nodeId : nodeIds) {
            const auto& node = nodes[nodeId];
            x_sum += node.position.x;
            y_sum += node.position.y;
            z_sum += node.position.z;
        }
        
        center.x = x_sum / nodeIds.size();
        center.y = y_sum / nodeIds.size();
        center.z = z_sum / nodeIds.size();
    }
    
    void calculateVolume(const std::vector<MeshNode<TNum>>& nodes) {
        // For hexahedral cell, approximate volume
        if (nodeIds.size() == 8) {
            // Simple approximation for regular hexahedron
            TNum dx = std::abs(nodes[nodeIds[1]].position.x - nodes[nodeIds[0]].position.x);
            TNum dy = std::abs(nodes[nodeIds[2]].position.y - nodes[nodeIds[0]].position.y);
            TNum dz = std::abs(nodes[nodeIds[4]].position.z - nodes[nodeIds[0]].position.z);
            volume = dx * dy * dz;
        } else {
            // Default to unit volume for other cell types
            volume = 1.0;
        }
    }
};

// Main mesh class
template <typename TNum>
class MeshObj {
private:
    std::vector<MeshNode<TNum>> nodes;
    std::vector<MeshCell<TNum>> cells;
    
    // Dimensions of the domain
    TNum xmin, xmax, ymin, ymax, zmin, zmax;
    
    // Base resolution
    int nx_base, ny_base, nz_base;
    
    // Maximum refinement level
    int max_level;
    
    // Refinement threshold
    TNum refinement_threshold;
    
    // Coarsening threshold
    TNum coarsening_threshold;
    
    // Refinement criterion
    RefinementCriterion criterion;
    
    // Custom refinement function
    std::function<TNum(const MeshCell<TNum>&)> custom_criterion_func;
    
    // Immersed boundary representation
    std::vector<std::function<bool(const Point3D<TNum>&)>> immersed_boundaries;
    
    // Map from (i,j,k,level) to cell index
    std::unordered_map<size_t, int> cell_map;
    
    // Hash function for (i,j,k,level)
    size_t hashCoords(int i, int j, int k, int level) const {
        // Simple hash function for coordinates and level
        return (static_cast<size_t>(i) << 40) | 
               (static_cast<size_t>(j) << 20) | 
               (static_cast<size_t>(k)) | 
               (static_cast<size_t>(level) << 60);
    }
    
    // Create a regular hexahedral cell
    int createHexCell(int i, int j, int k, int level) {
        // Calculate cell size at this level
        TNum h = std::pow(0.5, level);
        TNum dx = (xmax - xmin) / nx_base * h;
        TNum dy = (ymax - ymin) / ny_base * h;
        TNum dz = (zmax - zmin) / nz_base * h;
        
        // Calculate base coordinates
        TNum x = xmin + i * dx;
        TNum y = ymin + j * dy;
        TNum z = zmin + k * dz;
        
        // Create 8 nodes for hexahedral cell
        std::vector<int> nodeIds(8);
        
        // Create or find nodes
        for (int local = 0; local < 8; local++) {
            TNum offset_x = (local & 1) ? dx : 0;
            TNum offset_y = (local & 2) ? dy : 0;
            TNum offset_z = (local & 4) ? dz : 0;
            
            // Check if node already exists
            bool found = false;
            for (size_t n = 0; n < nodes.size(); n++) {
                if (std::abs(nodes[n].position.x - (x + offset_x)) < 1e-10 &&
                    std::abs(nodes[n].position.y - (y + offset_y)) < 1e-10 &&
                    std::abs(nodes[n].position.z - (z + offset_z)) < 1e-10) {
                    nodeIds[local] = n;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                MeshNode<TNum> node(nodes.size(), x + offset_x, y + offset_y, z + offset_z);
                nodes.push_back(node);
                nodeIds[local] = node.id;
            }
        }
        
        // Create cell
        MeshCell<TNum> cell(cells.size(), nodeIds, level);
        
        // Calculate cell center and volume
        cell.calculateCenter(nodes);
        cell.calculateVolume(nodes);
        
        // Check if cell intersects with immersed boundaries
        for (const auto& boundary_func : immersed_boundaries) {
            if (boundary_func(cell.center)) {
                cell.type = CellType::SOLID;
                break;
            }
        }
        
        // Add cell to nodes
        for (int nodeId : nodeIds) {
            nodes[nodeId].addCell(cell.id);
        }
        
        cells.push_back(cell);
        
        // Add to cell map
        cell_map[hashCoords(i, j, k, level)] = cell.id;
        
        return cell.id;
    }
    
    // Find neighboring cells
    void findNeighbors() {
        // Clear existing neighbors
        for (auto& cell : cells) {
            cell.neighborCells.clear();
        }
        
        // Build node-to-cell map for faster lookup
        std::vector<std::vector<int>> node_to_cells(nodes.size());
        for (size_t c = 0; c < cells.size(); c++) {
            for (int nodeId : cells[c].nodeIds) {
                node_to_cells[nodeId].push_back(c);
            }
        }
        
        // Find neighbors through shared nodes
        for (auto& cell : cells) {
            std::unordered_map<int, int> cell_count;
            
            // Count occurrences of cells connected to this cell's nodes
            for (int nodeId : cell.nodeIds) {
                for (int neighborCellId : node_to_cells[nodeId]) {
                    if (neighborCellId != cell.id) {
                        cell_count[neighborCellId]++;
                    }
                }
            }
            
            // Cells sharing a face have 4 common nodes in 3D hexahedral mesh
            for (const auto& pair : cell_count) {
                if (pair.second >= 4) {
                    cell.addNeighbor(pair.first);
                }
            }
        }
    }
    
    // Refine a cell
    void refineCell(int cellId) {
        const auto& parent = cells[cellId];
        
        // Only refine if below max level
        if (parent.level >= max_level) return;
        
        // Get parent cell indices
        int i_parent = 0, j_parent = 0, k_parent = 0, level_parent = parent.level;
        
        // Find parent cell in cell_map
        for (const auto& pair : cell_map) {
            if (pair.second == cellId) {
                size_t hash = pair.first;
                i_parent = (hash >> 40) & 0xFFFFF;
                j_parent = (hash >> 20) & 0xFFFFF;
                k_parent = hash & 0xFFFFF;
                break;
            }
        }
        
        // Create 8 child cells
        int new_level = level_parent + 1;
        for (int local = 0; local < 8; local++) {
            int i_offset = (local & 1) ? 1 : 0;
            int j_offset = (local & 2) ? 1 : 0;
            int k_offset = (local & 4) ? 1 : 0;
            
            int i_child = i_parent * 2 + i_offset;
            int j_child = j_parent * 2 + j_offset;
            int k_child = k_parent * 2 + k_offset;
            
            createHexCell(i_child, j_child, k_child, new_level);
        }
        
        // Mark parent cell as inactive (could be removed or kept for hierarchy)
        // For simplicity, we'll keep it but mark it
        cells[cellId].type = CellType::GHOST;
    }
    
    // Apply refinement based on criterion
    void applyRefinement() {
        std::vector<int> cells_to_refine;
        
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].type == CellType::GHOST) continue;
            
            TNum criterion_value = 0.0;
            
            switch (criterion) {
                case RefinementCriterion::GRADIENT: {
                    // Compute pressure gradient magnitude
                    TNum grad_p = 0.0;
                    for (int neighborId : cells[i].neighborCells) {
                        TNum dp = std::abs(cells[neighborId].p - cells[i].p);
                        grad_p = std::max(grad_p, dp);
                    }
                    criterion_value = grad_p;
                    break;
                }
                case RefinementCriterion::VORTICITY: {
                    // Approximate vorticity magnitude
                    TNum vort_x = 0.0, vort_y = 0.0, vort_z = 0.0;
                    
                    // Simple central difference for derivatives
                    for (int neighborId : cells[i].neighborCells) {
                        const auto& neighbor = cells[neighborId];
                        Vector3D<TNum> dir(
                            neighbor.center.x - cells[i].center.x,
                            neighbor.center.y - cells[i].center.y,
                            neighbor.center.z - cells[i].center.z
                        );
                        
                        // Normalize direction vector
                        TNum dist = dir.magnitude();
                        if (dist > 1e-10) {
                            dir = dir * (1.0 / dist);
                            
                            // Project velocity difference onto direction
                            TNum du = (neighbor.u - cells[i].u) / dist;
                            TNum dv = (neighbor.v - cells[i].v) / dist;
                            TNum dw = (neighbor.w - cells[i].w) / dist;
                            
                            // Accumulate vorticity components
                            vort_x += dir.y * dw - dir.z * dv;
                            vort_y += dir.z * du - dir.x * dw;
                            vort_z += dir.x * dv - dir.y * du;
                        }
                    }
                    
                    criterion_value = std::sqrt(vort_x*vort_x + vort_y*vort_y + vort_z*vort_z);
                    break;
                }
                case RefinementCriterion::CUSTOM: {
                    if (custom_criterion_func) {
                        criterion_value = custom_criterion_func(cells[i]);
                    }
                    break;
                }
            }
            
            // Check if cell needs refinement
            if (criterion_value > refinement_threshold) {
                cells_to_refine.push_back(i);
            }
        }
        
        // Refine cells
        for (int cellId : cells_to_refine) {
            refineCell(cellId);
        }
        
        // Update neighbors after refinement
        findNeighbors();
    }
    
    // Identify cells at immersed boundaries
    void identifyBoundaryCells() {
        for (auto& cell : cells) {
            if (cell.type == CellType::GHOST) continue;
            
            // Check if cell is a fluid cell with at least one solid neighbor
            if (cell.type == CellType::FLUID) {
                for (int neighborId : cell.neighborCells) {
                    if (cells[neighborId].type == CellType::SOLID) {
                        cell.type = CellType::BOUNDARY;
                        break;
                    }
                }
            }
        }
    }
    
    // Interpolate values at hanging nodes (nodes at refinement level transitions)
    void interpolateHangingNodes() {
        // Identify hanging nodes (nodes that connect cells of different refinement levels)
        std::unordered_map<int, std::vector<int>> hanging_nodes; // node_id -> [parent_cell_ids]
        
        for (const auto& cell : cells) {
            if (cell.type == CellType::GHOST) continue;
            
            for (int nodeId : cell.nodeIds) {
                for (int connectedCellId : nodes[nodeId].connectedCells) {
                    if (cells[connectedCellId].level != cell.level) {
                        hanging_nodes[nodeId].push_back(cell.id);
                    }
                }
            }
        }
        
        // For each hanging node, interpolate values from parent cells
        // This is a simplified approach - a more sophisticated method would be needed for production code
        for (const auto& pair : hanging_nodes) {
            int nodeId = pair.first;
            const auto& parentCellIds = pair.second;
            
            // Skip if no parent cells
            if (parentCellIds.empty()) continue;
            
            // Interpolate values at node from parent cells
            // For simplicity, we'll use inverse distance weighting
            TNum total_weight = 0.0;
            TNum weighted_u = 0.0, weighted_v = 0.0, weighted_w = 0.0, weighted_p = 0.0;
            
            for (int cellId : parentCellIds) {
                const auto& cell = cells[cellId];
                TNum dist = nodes[nodeId].position.distance(cell.center);
                
                // Avoid division by zero
                if (dist < 1e-10) dist = 1e-10;
                
                TNum weight = 1.0 / dist;
                total_weight += weight;
                
                weighted_u += cell.u * weight;
                weighted_v += cell.v * weight;
                weighted_w += cell.w * weight;
                weighted_p += cell.p * weight;
            }
            
            // Normalize weights
            if (total_weight > 0) {
                // Store interpolated values (in a real implementation, you'd store these at nodes or faces)
                // For this example, we'll just calculate them when needed
            }
        }
    }
    
    // Calculate fluxes at cell interfaces
    void calculateFluxes(std::vector<TNum>& flux_u, std::vector<TNum>& flux_v, std::vector<TNum>& flux_w) {
        // Resize flux vectors
        size_t num_interfaces = 0;
        for (const auto& cell : cells) {
            if (cell.type != CellType::GHOST) {
                num_interfaces += cell.neighborCells.size();
            }
        }
        
        flux_u.resize(num_interfaces, 0.0);
        flux_v.resize(num_interfaces, 0.0);
        flux_w.resize(num_interfaces, 0.0);
        
        // Map to store interface indices
        std::unordered_map<size_t, size_t> interface_map;
        size_t interface_count = 0;
        
        // Calculate fluxes at each interface
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].type == CellType::GHOST) continue;
            
            for (int neighborId : cells[i].neighborCells) {
                // Create a unique key for this interface
                size_t interface_key = (i < neighborId) ? 
                    (static_cast<size_t>(i) << 32) | neighborId :
                    (static_cast<size_t>(neighborId) << 32) | i;
                
                // Check if we've already processed this interface
                if (interface_map.find(interface_key) != interface_map.end()) {
                    continue;
                }
                
                // Store interface index
                interface_map[interface_key] = interface_count;
                
                // Calculate interface center
                Point3D<TNum> interface_center(
                    (cells[i].center.x + cells[neighborId].center.x) * 0.5,
                    (cells[i].center.y + cells[neighborId].center.y) * 0.5,
                    (cells[i].center.z + cells[neighborId].center.z) * 0.5
                );
                
                // Calculate normal vector from cell i to neighbor
                Vector3D<TNum> normal(
                    cells[neighborId].center.x - cells[i].center.x,
                    cells[neighborId].center.y - cells[i].center.y,
                    cells[neighborId].center.z - cells[i].center.z
                );
                normal = normal.normalize();
                
                // Interpolate velocities at interface
                TNum u_interface = (cells[i].u + cells[neighborId].u) * 0.5;
                TNum v_interface = (cells[i].v + cells[neighborId].v) * 0.5;
                TNum w_interface = (cells[i].w + cells[neighborId].w) * 0.5;
                
                // Calculate fluxes (simplified)
                flux_u[interface_count] = u_interface * normal.x;
                flux_v[interface_count] = v_interface * normal.y;
                flux_w[interface_count] = w_interface * normal.z;
                
                interface_count++;
            }
        }
    }
    
    // Apply immersed boundary method
    void applyImmersedBoundaryMethod() {
        // For each boundary cell, apply immersed boundary conditions
        for (auto& cell : cells) {
            if (cell.type != CellType::BOUNDARY) continue;
            
            // Find closest point on immersed boundary
            Point3D<TNum> closest_point = cell.center;
            TNum min_distance = std::numeric_limits<TNum>::max();
            
            // For each immersed boundary
            for (const auto& boundary_func : immersed_boundaries) {
                // In a real implementation, you would have a way to find the closest point
                // on the boundary. Here we'll use a simplified approach.
                
                // For demonstration, we'll just use the cell center if it's inside the boundary
                if (boundary_func(cell.center)) {
                    // This is a very simplified approach
                    // A real implementation would find the actual closest point on the boundary
                    closest_point = cell.center;
                    min_distance = 0.0;
                    break;
                }
            }
            
            // If we found a close boundary point, apply boundary conditions
            if (min_distance < std::numeric_limits<TNum>::max()) {
                // Apply no-slip condition (u=v=w=0 at boundary)
                // Interpolate between fluid and boundary
                // This is a simplified implementation
                
                // For each solid neighbor
                for (int neighborId : cell.neighborCells) {
                    if (cells[neighborId].type == CellType::SOLID) {
                        // Calculate distance to solid cell
                        TNum dist = cell.center.distance(cells[neighborId].center);
                        
                        // Apply linear interpolation for velocity
                        // Assuming zero velocity at solid boundary
                        TNum factor = dist / (dist + min_distance);
                        
                        // Adjust velocity in boundary cell
                        cell.u *= factor;
                        cell.v *= factor;
                        cell.w *= factor;
                    }
                }
            }
        }
    }

public:
    // Constructor
    MeshObj(TNum xmin_, TNum xmax_, TNum ymin_, TNum ymax_, TNum zmin_, TNum zmax_,
            int nx_, int ny_, int nz_, int max_level_ = 3)
        : xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), zmin(zmin_), zmax(zmax_),
          nx_base(nx_), ny_base(ny_), nz_base(nz_), max_level(max_level_),
          refinement_threshold(0.1), coarsening_threshold(0.01),
          criterion(RefinementCriterion::GRADIENT) {
        
        // Initialize base mesh
        initializeBaseMesh();
    }
    
    // Initialize base mesh
    void initializeBaseMesh() {
        // Clear existing mesh
        nodes.clear();
        cells.clear();
        cell_map.clear();
        
        // Create base level cells
        for (int k = 0; k < nz_base; k++) {
            for (int j = 0; j < ny_base; j++) {
                for (int i = 0; i < nx_base; i++) {
                    createHexCell(i, j, k, 0);
                }
            }
        }
        
        // Find neighbors
        findNeighbors();
    }
    
    // Add immersed boundary
    void addImmersedBoundary(std::function<bool(const Point3D<TNum>&)> boundary_func) {
        immersed_boundaries.push_back(boundary_func);
        
        // Update cell types based on new boundary
        for (auto& cell : cells) {
            if (cell.type != CellType::GHOST && boundary_func(cell.center)) {
                cell.type = CellType::SOLID;
            }
        }
        
        // Identify boundary cells
        identifyBoundaryCells();
    }
    
    // Set refinement criterion
    void setRefinementCriterion(RefinementCriterion criterion_, TNum threshold) {
        criterion = criterion_;
        refinement_threshold = threshold;
    }
    
    // Set custom refinement criterion
    void setCustomRefinementCriterion(std::function<TNum(const MeshCell<TNum>&)> criterion_func, TNum threshold) {
        criterion = RefinementCriterion::CUSTOM;
        custom_criterion_func = criterion_func;
        refinement_threshold = threshold;
    }
    
    // Adapt mesh based on current solution
    void adaptMesh() {
        applyRefinement();
        identifyBoundaryCells();
        interpolateHangingNodes();
    }
    
    // Get active cells (non-ghost cells)
    std::vector<int> getActiveCells() const {
        std::vector<int> active_cells;
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].type != CellType::GHOST) {
                active_cells.push_back(i);
            }
        }
        return active_cells;
    }
    
    // Get fluid cells
    std::vector<int> getFluidCells() const {
        std::vector<int> fluid_cells;
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].type == CellType::FLUID) {
                fluid_cells.push_back(i);
            }
        }
        return fluid_cells;
    }
    
    // Get boundary cells
    std::vector<int> getBoundaryCells() const {
        std::vector<int> boundary_cells;
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].type == CellType::BOUNDARY) {
                boundary_cells.push_back(i);
            }
        }
        return boundary_cells;
    }
    
    // Get solid cells
    std::vector<int> getSolidCells() const {
        std::vector<int> solid_cells;
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].type == CellType::SOLID) {
                solid_cells.push_back(i);
            }
        }
        return solid_cells;
    }
    
    // Get cell by ID
    const MeshCell<TNum>& getCell(int cellId) const {
        return cells[cellId];
    }
    
    // Get node by ID
    const MeshNode<TNum>& getNode(int nodeId) const {
        return nodes[nodeId];
    }
    
    // Get total number of cells
    size_t getNumCells() const {
        return cells.size();
    }
    
    // Get total number of nodes
    size_t getNumNodes() const {
        return nodes.size();
    }
    
    // Set field values for a cell
    void setCellValues(int cellId, TNum u_, TNum v_, TNum w_, TNum p_) {
        if (cellId >= 0 && cellId < static_cast<int>(cells.size())) {
            cells[cellId].u = u_;
            cells[cellId].v = v_;
            cells[cellId].w = w_;
            cells[cellId].p = p_;
        }
    }
    
    // Get field values for a cell
    void getCellValues(int cellId, TNum& u_, TNum& v_, TNum& w_, TNum& p_) const {
        if (cellId >= 0 && cellId < static_cast<int>(cells.size())) {
            u_ = cells[cellId].u;
            v_ = cells[cellId].v;
            w_ = cells[cellId].w;
            p_ = cells[cellId].p;
        }
    }
    
    // Export mesh to VTK format
    void exportToVTK(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }
        
        // Write VTK header
        file << "# vtk DataFile Version 3.0\n";
        file << "Adaptive Mesh\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";
        
        // Write points
        file << "POINTS " << nodes.size() << " float\n";
        for (const auto& node : nodes) {
            file << node.position.x << " " << node.position.y << " " << node.position.z << "\n";
        }
        
        // Write cells
        size_t active_cells = 0;
        for (const auto& cell : cells) {
            if (cell.type != CellType::GHOST) {
                active_cells++;
            }
        }
        
        file << "CELLS " << active_cells << " " << active_cells * 9 << "\n";
        for (const auto& cell : cells) {
            if (cell.type != CellType::GHOST) {
                file << "8";
                for (int nodeId : cell.nodeIds) {
                    file << " " << nodeId;
                }
                file << "\n";
            }
        }
        
        // Write cell types
        file << "CELL_TYPES " << active_cells << "\n";
        for (const auto& cell : cells) {
            if (cell.type != CellType::GHOST) {
                file << "12\n"; // 12 = VTK_HEXAHEDRON
            }
        }
        
        // Write cell data
        file << "CELL_DATA " << active_cells << "\n";
        
        // Write velocity
        file << "VECTORS velocity float\n";
        for (const auto& cell : cells) {
            if (cell.type != CellType::GHOST) {
                file << cell.u << " " << cell.v << " " << cell.w << "\n";
            }
        }
        
        // Write pressure
        file << "SCALARS pressure float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (const auto& cell : cells) {
            if (cell.type != CellType::GHOST) {
                file << cell.p << "\n";
            }
        }
        
        // Write cell type as scalar
        file << "SCALARS cell_type int 1\n";
        file << "LOOKUP_TABLE default\n";
        for (const auto& cell : cells) {
            if (cell.type != CellType::GHOST) {
                file << static_cast<int>(cell.type) << "\n";
            }
        }
        
        // Write refinement level
        file << "SCALARS level int 1\n";
        file << "LOOKUP_TABLE default\n";
        for (const auto& cell : cells) {
            if (cell.type != CellType::GHOST) {
                file << cell.level << "\n";
            }
        }
        
        file.close();
    }
    
    // Solve Poisson equation for pressure
    void solvePressurePoisson(int max_iterations = 1000, TNum tolerance = 1e-6) {
        std::vector<TNum> p_new(cells.size());
        
        // Initialize pressure
        for (size_t i = 0; i < cells.size(); i++) {
            p_new[i] = cells[i].p;
        }
        
        // Iterative solver (Jacobi method)
        for (int iter = 0; iter < max_iterations; iter++) {
            TNum max_residual = 0.0;
            
            // Update pressure for each cell
            for (size_t i = 0; i < cells.size(); i++) {
                if (cells[i].type == CellType::GHOST || cells[i].type == CellType::SOLID) {
                    continue;
                }
                
                // Calculate average pressure from neighbors
                TNum p_sum = 0.0;
                int num_fluid_neighbors = 0;
                
                for (int neighborId : cells[i].neighborCells) {
                    if (cells[neighborId].type != CellType::SOLID && cells[neighborId].type != CellType::GHOST) {
                        p_sum += p_new[neighborId];
                        num_fluid_neighbors++;
                    }
                }
                
                if (num_fluid_neighbors > 0) {
                    TNum p_old = p_new[i];
                    p_new[i] = p_sum / num_fluid_neighbors;
                    
                    // Calculate residual
                    TNum residual = std::abs(p_new[i] - p_old);
                    max_residual = std::max(max_residual, residual);
                }
            }
            
            // Update pressure in cells
            for (size_t i = 0; i < cells.size(); i++) {
                cells[i].p = p_new[i];
            }
            
            // Check convergence
            if (max_residual < tolerance) {
                break;
            }
        }
    }
    
    // Project velocity field to be divergence-free
    void projectVelocityField() {
        // Calculate divergence at each cell
        std::vector<TNum> divergence(cells.size(), 0.0);
        
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].type == CellType::GHOST || cells[i].type == CellType::SOLID) {
                continue;
            }
            
            TNum div = 0.0;
            
            for (int neighborId : cells[i].neighborCells) {
                if (cells[neighborId].type != CellType::SOLID && cells[neighborId].type != CellType::GHOST) {
                    // Calculate normal vector from cell i to neighbor
                    Vector3D<TNum> normal(
                        cells[neighborId].center.x - cells[i].center.x,
                        cells[neighborId].center.y - cells[i].center.y,
                        cells[neighborId].center.z - cells[i].center.z
                    );
                    
                    TNum dist = normal.magnitude();
                    if (dist > 1e-10) {
                        normal = normal * (1.0 / dist);
                        
                        // Calculate velocity at interface
                        TNum u_interface = (cells[i].u + cells[neighborId].u) * 0.5;
                        TNum v_interface = (cells[i].v + cells[neighborId].v) * 0.5;
                        TNum w_interface = (cells[i].w + cells[neighborId].w) * 0.5;
                        
                        // Calculate flux through interface
                        TNum flux = u_interface * normal.x + v_interface * normal.y + w_interface * normal.z;
                        
                        // Accumulate divergence
                        div += flux;
                    }
                }
            }
            
            divergence[i] = div / cells[i].volume;
        }
        
        // Solve Poisson equation for pressure correction
        // (Using the pressure solver we already have)
        solvePressurePoisson();
        
        // Correct velocities
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].type == CellType::GHOST || cells[i].type == CellType::SOLID) {
                continue;
            }
            
            for (int neighborId : cells[i].neighborCells) {
                if (cells[neighborId].type != CellType::SOLID && cells[neighborId].type != CellType::GHOST) {
                    // Calculate pressure gradient
                    TNum dp = cells[neighborId].p - cells[i].p;
                    
                    // Calculate normal vector from cell i to neighbor
                    Vector3D<TNum> normal(
                        cells[neighborId].center.x - cells[i].center.x,
                        cells[neighborId].center.y - cells[i].center.y,
                        cells[neighborId].center.z - cells[i].center.z
                    );
                    
                    TNum dist = normal.magnitude();
                    if (dist > 1e-10) {
                        normal = normal * (1.0 / dist);
                        
                        // Correct velocity components
                        cells[i].u -= dp * normal.x;
                        cells[i].v -= dp * normal.y;
                        cells[i].w -= dp * normal.z;
                    }
                }
            }
        }
    }
    
    // Apply boundary conditions
    void applyBoundaryConditions() {
        // Apply immersed boundary conditions
        applyImmersedBoundaryMethod();
        
        // Apply other boundary conditions as needed
        // For example, enforce no-slip at domain boundaries
        for (auto& cell : cells) {
            if (cell.type == CellType::GHOST) continue;
            
            // Check if cell is at domain boundary
            bool at_boundary = false;
            for (int nodeId : cell.nodeIds) {
                const auto& node = nodes[nodeId];
                if (std::abs(node.position.x - xmin) < 1e-10 || std::abs(node.position.x - xmax) < 1e-10 ||
                    std::abs(node.position.y - ymin) < 1e-10 || std::abs(node.position.y - ymax) < 1e-10 ||
                    std::abs(node.position.z - zmin) < 1e-10 || std::abs(node.position.z - zmax) < 1e-10) {
                    at_boundary = true;
                    break;
                }
            }
            
            if (at_boundary) {
                // Apply no-slip condition at domain boundary
                cell.u = 0.0;
                cell.v = 0.0;
                cell.w = 0.0;
            }
        }
    }
    
    // Advance solution one time step
    void advanceTimeStep(TNum dt) {
        // Store old velocities
        std::vector<TNum> u_old(cells.size());
        std::vector<TNum> v_old(cells.size());
        std::vector<TNum> w_old(cells.size());
        
        for (size_t i = 0; i < cells.size(); i++) {
            u_old[i] = cells[i].u;
            v_old[i] = cells[i].v;
            w_old[i] = cells[i].w;
        }
        
        // Calculate fluxes
        std::vector<TNum> flux_u, flux_v, flux_w;
        calculateFluxes(flux_u, flux_v, flux_w);
        
        // Update velocities (simplified advection-diffusion)
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].type == CellType::GHOST || cells[i].type == CellType::SOLID) {
                continue;
            }
            
            // Advection term (simplified)
            TNum advection_u = 0.0, advection_v = 0.0, advection_w = 0.0;
            
            // Diffusion term (simplified)
            TNum diffusion_u = 0.0, diffusion_v = 0.0, diffusion_w = 0.0;
            int num_neighbors = 0;
            
            for (int neighborId : cells[i].neighborCells) {
                if (cells[neighborId].type != CellType::SOLID && cells[neighborId].type != CellType::GHOST) {
                    // Calculate normal vector from cell i to neighbor
                    Vector3D<TNum> normal(
                        cells[neighborId].center.x - cells[i].center.x,
                        cells[neighborId].center.y - cells[i].center.y,
                        cells[neighborId].center.z - cells[i].center.z
                    );
                    
                    TNum dist = normal.magnitude();
                    if (dist > 1e-10) {
                        // Diffusion term (Laplacian approximation)
                        diffusion_u += (u_old[neighborId] - u_old[i]) / (dist * dist);
                        diffusion_v += (v_old[neighborId] - v_old[i]) / (dist * dist);
                        diffusion_w += (w_old[neighborId] - w_old[i]) / (dist * dist);
                        num_neighbors++;
                    }
                }
            }
            
            // Normalize diffusion term
            if (num_neighbors > 0) {
                diffusion_u /= num_neighbors;
                diffusion_v /= num_neighbors;
                diffusion_w /= num_neighbors;
            }
            
            // Update velocities
            TNum viscosity = 0.01; // Example viscosity coefficient
            cells[i].u = u_old[i] + dt * (viscosity * diffusion_u - advection_u);
            cells[i].v = v_old[i] + dt * (viscosity * diffusion_v - advection_v);
            cells[i].w = w_old[i] + dt * (viscosity * diffusion_w - advection_w);
        }
        
        // Project velocity field to be divergence-free
        projectVelocityField();
        
        // Apply boundary conditions
        applyBoundaryConditions();
        
        // Adapt mesh based on solution
        adaptMesh();
    }
    
    // Run simulation for specified number of time steps
    void runSimulation(int num_steps, TNum dt, const std::string& output_prefix = "output", int output_frequency = 10) {
        for (int step = 0; step < num_steps; step++) {
            // Advance solution
            advanceTimeStep(dt);
            
            // Output results at specified frequency
            if (step % output_frequency == 0) {
                std::string filename = output_prefix + "_" + std::to_string(step) + ".vtk";
                exportToVTK(filename);
            }
        }
    }
};

#endif // MESH_OBJ_HPP