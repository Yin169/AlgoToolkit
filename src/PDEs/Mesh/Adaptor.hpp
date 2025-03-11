#ifndef MESH_ADAPTER_HPP
#define MESH_ADAPTER_HPP

#include "../../Obj/VectorObj.hpp"
#include <vector>
#include <array>
#include <functional>
#include <cmath>
#include <algorithm>
#include <memory>

// Forward declaration
template <typename TNum>
class NavierStokesSolver3D;

/**
 * @brief Enum for different mesh types
 */
enum class MeshType {
    CARTESIAN,    // Regular Cartesian grid
    CYLINDRICAL,  // Cylindrical coordinates (r, θ, z)
    SPHERICAL,    // Spherical coordinates (r, θ, φ)
    CURVILINEAR   // General curvilinear coordinates
};

/**
 * @brief Class to adapt arbitrary meshes to the structured grid used by NavierStokesSolver3D
 */
template <typename TNum>
class MeshAdapter {
private:
    // Mesh dimensions
    int nx, ny, nz;
    
    // Mesh type
    MeshType mesh_type;
    
    // Coordinate transformation functions
    std::function<std::array<TNum, 3>(TNum, TNum, TNum)> physical_to_computational;
    std::function<std::array<TNum, 3>(TNum, TNum, TNum)> computational_to_physical;
    
    // Jacobian matrix and its determinant
    std::vector<std::array<TNum, 9>> jacobian_matrix;  // 3x3 matrix at each point
    std::vector<TNum> jacobian_determinant;
    
    // Metric tensors for curvilinear coordinates
    std::vector<std::array<TNum, 9>> metric_tensor;    // Covariant metric tensor
    std::vector<std::array<TNum, 9>> metric_tensor_inv; // Contravariant metric tensor
    
    // Christoffel symbols for curvilinear coordinates (only used for CURVILINEAR type)
    std::vector<std::array<TNum, 27>> christoffel_symbols; // 3x3x3 tensor at each point
    
    // Grid spacing in computational domain
    TNum dx_comp, dy_comp, dz_comp;
    
    // Physical domain bounds
    TNum x_min, x_max, y_min, y_max, z_min, z_max;
    
    // Compute index in flattened array
    int idx(int i, int j, int k) const {
        return i + j * nx + k * nx * ny;
    }
    
    // Compute Jacobian matrix at a point
    std::array<TNum, 9> computeJacobianMatrix(TNum x_comp, TNum y_comp, TNum z_comp) const {
        // Small delta for finite difference approximation
        const TNum delta = 1e-6;
        
        // Compute physical coordinates at the point
        auto p0 = computational_to_physical(x_comp, y_comp, z_comp);
        
        // Compute physical coordinates at x_comp + delta
        auto px = computational_to_physical(x_comp + delta, y_comp, z_comp);
        
        // Compute physical coordinates at y_comp + delta
        auto py = computational_to_physical(x_comp, y_comp + delta, z_comp);
        
        // Compute physical coordinates at z_comp + delta
        auto pz = computational_to_physical(x_comp, y_comp, z_comp + delta);
        
        // Compute partial derivatives using finite differences
        TNum dx_dxi = (px[0] - p0[0]) / delta;
        TNum dy_dxi = (px[1] - p0[1]) / delta;
        TNum dz_dxi = (px[2] - p0[2]) / delta;
        
        TNum dx_deta = (py[0] - p0[0]) / delta;
        TNum dy_deta = (py[1] - p0[1]) / delta;
        TNum dz_deta = (py[2] - p0[2]) / delta;
        
        TNum dx_dzeta = (pz[0] - p0[0]) / delta;
        TNum dy_dzeta = (pz[1] - p0[1]) / delta;
        TNum dz_dzeta = (pz[2] - p0[2]) / delta;
        
        // Return Jacobian matrix as a flattened array
        return {
            dx_dxi, dy_dxi, dz_dxi,
            dx_deta, dy_deta, dz_deta,
            dx_dzeta, dy_dzeta, dz_dzeta
        };
    }
    
    // Compute determinant of a 3x3 matrix
    TNum computeDeterminant(const std::array<TNum, 9>& matrix) const {
        return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
               matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
               matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    }
    
    // Compute inverse of a 3x3 matrix
    std::array<TNum, 9> computeInverse(const std::array<TNum, 9>& matrix, TNum det) const {
        TNum inv_det = 1.0 / det;
        
        std::array<TNum, 9> inverse;
        
        inverse[0] = inv_det * (matrix[4] * matrix[8] - matrix[5] * matrix[7]);
        inverse[1] = inv_det * (matrix[2] * matrix[7] - matrix[1] * matrix[8]);
        inverse[2] = inv_det * (matrix[1] * matrix[5] - matrix[2] * matrix[4]);
        
        inverse[3] = inv_det * (matrix[5] * matrix[6] - matrix[3] * matrix[8]);
        inverse[4] = inv_det * (matrix[0] * matrix[8] - matrix[2] * matrix[6]);
        inverse[5] = inv_det * (matrix[2] * matrix[3] - matrix[0] * matrix[5]);
        
        inverse[6] = inv_det * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
        inverse[7] = inv_det * (matrix[1] * matrix[6] - matrix[0] * matrix[7]);
        inverse[8] = inv_det * (matrix[0] * matrix[4] - matrix[1] * matrix[3]);
        
        return inverse;
    }
    
    // Compute metric tensor from Jacobian matrix
    std::array<TNum, 9> computeMetricTensor(const std::array<TNum, 9>& jacobian) const {
        std::array<TNum, 9> metric;
        
        // g_ij = Σ_k (∂x_k/∂ξ_i) * (∂x_k/∂ξ_j)
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                metric[i*3 + j] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    metric[i*3 + j] += jacobian[i*3 + k] * jacobian[j*3 + k];
                }
            }
        }
        
        return metric;
    }
    
    // Compute Christoffel symbols for curvilinear coordinates
    std::array<TNum, 27> computeChristoffelSymbols(int i, int j, int k) const {
        std::array<TNum, 27> symbols;
        std::fill(symbols.begin(), symbols.end(), 0.0);
        
        // For simple mesh types, we can use analytical formulas
        if (mesh_type == MeshType::CYLINDRICAL) {
            // Get physical coordinates (r, θ, z)
            TNum x_comp = i * dx_comp;
            TNum y_comp = j * dy_comp;
            TNum z_comp = k * dz_comp;
            
            auto phys = computational_to_physical(x_comp, y_comp, z_comp);
            TNum r = std::sqrt(phys[0]*phys[0] + phys[1]*phys[1]); // Get r from x,y
            
            // Non-zero Christoffel symbols for cylindrical coordinates
            // Γ^r_θθ = -r
            symbols[0*9 + 1*3 + 1] = -r;
            
            // Γ^θ_rθ = Γ^θ_θr = 1/r
            symbols[1*9 + 0*3 + 1] = 1.0/r;
            symbols[1*9 + 1*3 + 0] = 1.0/r;
        }
        else if (mesh_type == MeshType::SPHERICAL) {
            // Get physical coordinates (r, θ, φ)
            TNum x_comp = i * dx_comp;
            TNum y_comp = j * dy_comp;
            TNum z_comp = k * dz_comp;
            
            auto phys = computational_to_physical(x_comp, y_comp, z_comp);
            TNum r = phys[0];
            TNum theta = phys[1];
            
            // Non-zero Christoffel symbols for spherical coordinates
            // Γ^r_θθ = -r
            symbols[0*9 + 1*3 + 1] = -r;
            
            // Γ^r_φφ = -r*sin²(θ)
            symbols[0*9 + 2*3 + 2] = -r * std::sin(theta) * std::sin(theta);
            
            // Γ^θ_rθ = Γ^θ_θr = 1/r
            symbols[1*9 + 0*3 + 1] = 1.0/r;
            symbols[1*9 + 1*3 + 0] = 1.0/r;
            
            // Γ^θ_φφ = -sin(θ)*cos(θ)
            symbols[1*9 + 2*3 + 2] = -std::sin(theta) * std::cos(theta);
            
            // Γ^φ_rφ = Γ^φ_φr = 1/r
            symbols[2*9 + 0*3 + 2] = 1.0/r;
            symbols[2*9 + 2*3 + 0] = 1.0/r;
            
            // Γ^φ_θφ = Γ^φ_φθ = cot(θ)
            symbols[2*9 + 1*3 + 2] = 1.0/std::tan(theta);
            symbols[2*9 + 2*3 + 1] = 1.0/std::tan(theta);
        }
        else if (mesh_type == MeshType::CURVILINEAR) {
            // For general curvilinear coordinates, compute numerically
            // This is a complex calculation that would require computing
            // derivatives of the metric tensor, which is beyond the scope
            // of this example
        }
        
        return symbols;
    }
    
    // Initialize coordinate transformations for different mesh types
    void initializeCoordinateTransformations() {
        switch (mesh_type) {
            case MeshType::CARTESIAN:
                // Identity transformation
                computational_to_physical = [this](TNum xi, TNum eta, TNum zeta) -> std::array<TNum, 3> {
                    TNum x = x_min + xi * (x_max - x_min) / (nx - 1);
                    TNum y = y_min + eta * (y_max - y_min) / (ny - 1);
                    TNum z = z_min + zeta * (z_max - z_min) / (nz - 1);
                    return {x, y, z};
                };
                
                physical_to_computational = [this](TNum x, TNum y, TNum z) -> std::array<TNum, 3> {
                    TNum xi = (x - x_min) * (nx - 1) / (x_max - x_min);
                    TNum eta = (y - y_min) * (ny - 1) / (y_max - y_min);
                    TNum zeta = (z - z_min) * (nz - 1) / (z_max - z_min);
                    return {xi, eta, zeta};
                };
                break;
                
                case MeshType::CYLINDRICAL:
                // Cylindrical coordinates (r, θ, z)
                computational_to_physical = [this](TNum xi, TNum eta, TNum zeta) -> std::array<TNum, 3> {
                    // Map computational coordinates [0,1] to physical domain
                    TNum r = x_min + xi * (x_max - x_min);
                    TNum theta = y_min + eta * (y_max - y_min);
                    TNum z = z_min + zeta * (z_max - z_min);
                    
                    // Convert to Cartesian coordinates
                    TNum x = r * std::cos(theta);
                    TNum y = r * std::sin(theta);
                    
                    return {x, y, z};
                };
                
                physical_to_computational = [this](TNum x, TNum y, TNum z) -> std::array<TNum, 3> {
                    // Convert from Cartesian to cylindrical
                    TNum r = std::sqrt(x*x + y*y);
                    TNum theta = std::atan2(y, x);
                    
                    // Normalize theta to [0, 2π)
                    if (theta < 0) theta += 2 * M_PI;
                    
                    // Map to computational domain [0,1]
                    TNum xi = (r - x_min) / (x_max - x_min);
                    TNum eta = (theta - y_min) / (y_max - y_min);
                    TNum zeta = (z - z_min) / (z_max - z_min);
                    
                    return {xi, eta, zeta};
                };
                break;
                
            case MeshType::SPHERICAL:
                // Spherical coordinates (r, θ, φ)
                computational_to_physical = [this](TNum xi, TNum eta, TNum zeta) -> std::array<TNum, 3> {
                    TNum r = x_min + xi * (x_max - x_min) / (nx - 1);
                    TNum theta = y_min + eta * (y_max - y_min) / (ny - 1);
                    TNum phi = z_min + zeta * (z_max - z_min) / (nz - 1);
                    
                    TNum x = r * std::sin(theta) * std::cos(phi);
                    TNum y = r * std::sin(theta) * std::sin(phi);
                    TNum z = r * std::cos(theta);
                    
                    return {x, y, z};
                };
                
                physical_to_computational = [this](TNum x, TNum y, TNum z) -> std::array<TNum, 3> {
                    TNum r = std::sqrt(x*x + y*y + z*z);
                    TNum theta = std::acos(z / r);
                    TNum phi = std::atan2(y, x);
                    
                    // Normalize phi to [0, 2π)
                    if (phi < 0) phi += 2 * M_PI;
                    
                    TNum xi = (r - x_min) * (nx - 1) / (x_max - x_min);
                    TNum eta = (theta - y_min) * (ny - 1) / (y_max - y_min);
                    TNum zeta = (phi - z_min) * (nz - 1) / (z_max - z_min);
                    
                    return {xi, eta, zeta};
                };
                break;
                
            case MeshType::CURVILINEAR:
                // For general curvilinear coordinates, the transformation functions
                // must be provided by the user through setCoordinateTransformations()
                break;
        }
    }
    
    // Precompute geometric quantities for the entire mesh
    void precomputeGeometricQuantities() {
        int n = nx * ny * nz;
        jacobian_matrix.resize(n);
        jacobian_determinant.resize(n);
        metric_tensor.resize(n);
        metric_tensor_inv.resize(n);
        
        if (mesh_type == MeshType::CURVILINEAR) {
            christoffel_symbols.resize(n);
        }
        
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int index = idx(i, j, k);
                    
                    // Compute Jacobian matrix
                    TNum x_comp = i * dx_comp;
                    TNum y_comp = j * dy_comp;
                    TNum z_comp = k * dz_comp;
                    
                    jacobian_matrix[index] = computeJacobianMatrix(x_comp, y_comp, z_comp);
                    
                    // Compute Jacobian determinant
                    jacobian_determinant[index] = computeDeterminant(jacobian_matrix[index]);
                    
                    // Compute metric tensor
                    metric_tensor[index] = computeMetricTensor(jacobian_matrix[index]);
                    
                    // Compute inverse metric tensor
                    metric_tensor_inv[index] = computeInverse(metric_tensor[index], 
                                                             computeDeterminant(metric_tensor[index]));
                    
                    // Compute Christoffel symbols for curvilinear coordinates
                    if (mesh_type == MeshType::CURVILINEAR) {
                        christoffel_symbols[index] = computeChristoffelSymbols(i, j, k);
                    }
                }
            }
        }
    }
    
    // Transform velocity components from physical to computational space
    void transformVelocity(const VectorObj<TNum>& u_phys, const VectorObj<TNum>& v_phys, 
                          const VectorObj<TNum>& w_phys, VectorObj<TNum>& u_comp, 
                          VectorObj<TNum>& v_comp, VectorObj<TNum>& w_comp) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int index = idx(i, j, k);
                    
                    // Get Jacobian matrix at this point
                    const auto& J = jacobian_matrix[index];
                    
                    // Transform velocity components: u_comp = J^T * u_phys
                    u_comp[index] = J[0] * u_phys[index] + J[1] * v_phys[index] + J[2] * w_phys[index];
                    v_comp[index] = J[3] * u_phys[index] + J[4] * v_phys[index] + J[5] * w_phys[index];
                    w_comp[index] = J[6] * u_phys[index] + J[7] * v_phys[index] + J[8] * w_phys[index];
                }
            }
        }
    }
    
    // Transform velocity components from computational to physical space
    void transformVelocityBack(const VectorObj<TNum>& u_comp, const VectorObj<TNum>& v_comp, 
                              const VectorObj<TNum>& w_comp, VectorObj<TNum>& u_phys, 
                              VectorObj<TNum>& v_phys, VectorObj<TNum>& w_phys) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int index = idx(i, j, k);
                    
                    // Get Jacobian matrix at this point
                    const auto& J = jacobian_matrix[index];
                    
                    // Compute inverse Jacobian
                    auto J_inv = computeInverse(J, jacobian_determinant[index]);
                    
                    // Transform velocity components: u_phys = J_inv * u_comp
                    u_phys[index] = J_inv[0] * u_comp[index] + J_inv[1] * v_comp[index] + J_inv[2] * w_comp[index];
                    v_phys[index] = J_inv[3] * u_comp[index] + J_inv[4] * v_comp[index] + J_inv[5] * w_comp[index];
                    w_phys[index] = J_inv[6] * u_comp[index] + J_inv[7] * v_comp[index] + J_inv[8] * w_comp[index];
                }
            }
        }
    }
    
    // Transform pressure gradient from physical to computational space
    void transformPressureGradient(const VectorObj<TNum>& dp_dx, const VectorObj<TNum>& dp_dy, 
                                  const VectorObj<TNum>& dp_dz, VectorObj<TNum>& dp_dxi, 
                                  VectorObj<TNum>& dp_deta, VectorObj<TNum>& dp_dzeta) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int index = idx(i, j, k);
                    
                    // Get Jacobian matrix at this point
                    const auto& J = jacobian_matrix[index];
                    
                    // Transform pressure gradient: dp_dcomp = J * dp_dphys
                    dp_dxi[index] = J[0] * dp_dx[index] + J[3] * dp_dy[index] + J[6] * dp_dz[index];
                    dp_deta[index] = J[1] * dp_dx[index] + J[4] * dp_dy[index] + J[7] * dp_dz[index];
                    dp_dzeta[index] = J[2] * dp_dx[index] + J[5] * dp_dy[index] + J[8] * dp_dz[index];
                }
            }
        }
    }

public:
    // Constructor for standard mesh types
    MeshAdapter(int nx_, int ny_, int nz_, MeshType mesh_type_,
               TNum x_min_, TNum x_max_, TNum y_min_, TNum y_max_, TNum z_min_, TNum z_max_)
        : nx(nx_), ny(ny_), nz(nz_), mesh_type(mesh_type_),
          x_min(x_min_), x_max(x_max_), y_min(y_min_), y_max(y_max_), z_min(z_min_), z_max(z_max_) {
        
        // Compute grid spacing in computational domain
        dx_comp = 1.0 / (nx - 1);
        dy_comp = 1.0 / (ny - 1);
        dz_comp = 1.0 / (nz - 1);
        
        // Initialize coordinate transformations
        initializeCoordinateTransformations();
        
        // Precompute geometric quantities
        precomputeGeometricQuantities();
    }
    
    // Set custom coordinate transformation functions for curvilinear meshes
    void setCoordinateTransformations(
        std::function<std::array<TNum, 3>(TNum, TNum, TNum)> comp_to_phys,
        std::function<std::array<TNum, 3>(TNum, TNum, TNum)> phys_to_comp) {
        
        computational_to_physical = comp_to_phys;
        physical_to_computational = phys_to_comp;
        
        // Recompute geometric quantities with the new transformations
        precomputeGeometricQuantities();
    }
    
    // Get mesh dimensions
    void getMeshDimensions(int& nx_, int& ny_, int& nz_) const {
        nx_ = nx;
        ny_ = ny;
        nz_ = nz;
    }
    
    // Get physical coordinates for a computational grid point
    std::array<TNum, 3> getPhysicalCoordinates(int i, int j, int k) const {
        TNum x_comp = i * dx_comp;
        TNum y_comp = j * dy_comp;
        TNum z_comp = k * dz_comp;
        
        return computational_to_physical(x_comp, y_comp, z_comp);
    }
    
    // Get computational coordinates for a physical point
    std::array<TNum, 3> getComputationalCoordinates(TNum x, TNum y, TNum z) const {
        return physical_to_computational(x, y, z);
    }
    
    // Get Jacobian determinant at a grid point
    TNum getJacobianDeterminant(int i, int j, int k) const {
        return jacobian_determinant[idx(i, j, k)];
    }
    
    // Adapt physical velocity field to computational grid for NavierStokesSolver3D
    void adaptVelocityField(const VectorObj<TNum>& u_phys, const VectorObj<TNum>& v_phys, 
                           const VectorObj<TNum>& w_phys, VectorObj<TNum>& u_comp, 
                           VectorObj<TNum>& v_comp, VectorObj<TNum>& w_comp) {
        // Resize computational velocity fields if needed
        int n = nx * ny * nz;
        if (u_comp.size() != n) u_comp.resize(n, 0.0);
        if (v_comp.size() != n) v_comp.resize(n, 0.0);
        if (w_comp.size() != n) w_comp.resize(n, 0.0);
        
        // Transform velocity from physical to computational space
        transformVelocity(u_phys, v_phys, w_phys, u_comp, v_comp, w_comp);
    }
    
    // Convert computational velocity back to physical space
    void convertVelocityToPhysical(const VectorObj<TNum>& u_comp, const VectorObj<TNum>& v_comp, 
                                  const VectorObj<TNum>& w_comp, VectorObj<TNum>& u_phys, 
                                  VectorObj<TNum>& v_phys, VectorObj<TNum>& w_phys) {
        // Resize physical velocity fields if needed
        int n = nx * ny * nz;
        if (u_phys.size() != n) u_phys.resize(n, 0.0);
        if (v_phys.size() != n) v_phys.resize(n, 0.0);
        if (w_phys.size() != n) w_phys.resize(n, 0.0);
        
        // Transform velocity from computational to physical space
        transformVelocityBack(u_comp, v_comp, w_comp, u_phys, v_phys, w_phys);
    }
    
    // Adapt boundary conditions from physical to computational space
    std::function<TNum(TNum, TNum, TNum, TNum)> adaptBoundaryCondition(
        std::function<TNum(TNum, TNum, TNum, TNum)> bc_phys, int component) {
        
        return [this, bc_phys, component](TNum xi, TNum eta, TNum zeta, TNum t) -> TNum {
            // Convert computational coordinates to physical
            auto phys_coords = computational_to_physical(xi, eta, zeta);
            
            // Get physical boundary condition value
            TNum bc_value = bc_phys(phys_coords[0], phys_coords[1], phys_coords[2], t);
            
            // For simple cases, we can just return the physical value
            // For more complex cases, we would need to transform the boundary condition
            // based on the component and the coordinate system
            return bc_value;
        };
    }
    
    // Create a NavierStokesSolver3D with the adapted mesh
    std::unique_ptr<NavierStokesSolver3D<TNum>> createSolver(TNum dt, TNum Re) {
        // Create solver with computational grid dimensions
        auto solver = std::make_unique<NavierStokesSolver3D<TNum>>(
            nx, ny, nz, dx_comp, dy_comp, dz_comp, dt, Re);
        
        return solver;
    }
    
    // Adapt external forces from physical to computational space
    void adaptExternalForces(const VectorObj<TNum>& fx_phys, const VectorObj<TNum>& fy_phys, 
                            const VectorObj<TNum>& fz_phys, VectorObj<TNum>& fx_comp, 
                            VectorObj<TNum>& fy_comp, VectorObj<TNum>& fz_comp) {
        // Resize computational force fields if needed
        int n = nx * ny * nz;
        if (fx_comp.size() != n) fx_comp.resize(n, 0.0);
        if (fy_comp.size() != n) fy_comp.resize(n, 0.0);
        if (fz_comp.size() != n) fz_comp.resize(n, 0.0);
        
        // Transform forces from physical to computational space (similar to velocity)
        transformVelocity(fx_phys, fy_phys, fz_phys, fx_comp, fy_comp, fz_comp);
    }
    
    // Get grid spacing in computational domain
    void getComputationalGridSpacing(TNum& dx_, TNum& dy_, TNum& dz_) const {
        dx_ = dx_comp;
        dy_ = dy_comp;
        dz_ = dz_comp;
    }
    
    // Get physical domain bounds
    void getPhysicalDomainBounds(TNum& x_min_, TNum& x_max_, TNum& y_min_, 
		TNum& y_max_, TNum& z_min_, TNum& z_max_) const {
			x_min_ = x_min;
			x_max_ = x_max;
			y_min_ = y_min;
			y_max_ = y_max;
			z_min_ = z_min;
			z_max_ = z_max;
		}
		
		// Get mesh type
		MeshType getMeshType() const {
			return mesh_type;
		}
		
		// Get metric tensor at a grid point
		std::array<TNum, 9> getMetricTensor(int i, int j, int k) const {
			return metric_tensor[idx(i, j, k)];
		}
		
		// Get inverse metric tensor at a grid point
		std::array<TNum, 9> getInverseMetricTensor(int i, int j, int k) const {
			return metric_tensor_inv[idx(i, j, k)];
		}
		
		// Get Christoffel symbols at a grid point (only for curvilinear coordinates)
		std::array<TNum, 27> getChristoffelSymbols(int i, int j, int k) const {
			if (mesh_type != MeshType::CURVILINEAR) {
				// For non-curvilinear meshes, return zeros
				std::array<TNum, 27> zeros;
				std::fill(zeros.begin(), zeros.end(), 0.0);
				return zeros;
			}
			return christoffel_symbols[idx(i, j, k)];
		}
		
		// Compute the divergence of a vector field in physical space
		VectorObj<TNum> computeDivergence(const VectorObj<TNum>& u_phys, 
										 const VectorObj<TNum>& v_phys, 
										 const VectorObj<TNum>& w_phys) const {
			int n = nx * ny * nz;
			VectorObj<TNum> div(n, 0.0);
			
			// For Cartesian coordinates, the divergence is simple
			if (mesh_type == MeshType::CARTESIAN) {
				// Compute physical grid spacing
				TNum dx_phys = (x_max - x_min) / (nx - 1);
				TNum dy_phys = (y_max - y_min) / (ny - 1);
				TNum dz_phys = (z_max - z_min) / (nz - 1);
				
				// Use central differences for interior points
				for (int k = 1; k < nz - 1; ++k) {
					for (int j = 1; j < ny - 1; ++j) {
						for (int i = 1; i < nx - 1; ++i) {
							int index = idx(i, j, k);
							
							// du/dx + dv/dy + dw/dz
							div[index] = (u_phys[idx(i+1, j, k)] - u_phys[idx(i-1, j, k)]) / (2 * dx_phys) +
										(v_phys[idx(i, j+1, k)] - v_phys[idx(i, j-1, k)]) / (2 * dy_phys) +
										(w_phys[idx(i, j, k+1)] - w_phys[idx(i, j, k-1)]) / (2 * dz_phys);
						}
					}
				}
			} else {
				// For non-Cartesian coordinates, use the general formula:
				// div(V) = (1/J) * sum_i d/dxi(J * V^i)
				// where J is the Jacobian determinant and V^i are contravariant components
				
				// First, transform velocity to computational space
				VectorObj<TNum> u_comp(n), v_comp(n), w_comp(n);
				
				// Use a const_cast here since transformVelocity is non-const but doesn't modify the inputs
				const_cast<MeshAdapter*>(this)->transformVelocity(u_phys, v_phys, w_phys, u_comp, v_comp, w_comp);
				
				// Compute divergence using the transformed velocity
				for (int k = 1; k < nz - 1; ++k) {
					for (int j = 1; j < ny - 1; ++j) {
						for (int i = 1; i < nx - 1; ++i) {
							int index = idx(i, j, k);
							
							// Get Jacobian determinant at this point
							TNum J = jacobian_determinant[index];
							
							// Compute d/dxi(J * u_comp), d/deta(J * v_comp), d/dzeta(J * w_comp)
							TNum d_Ju_dxi = (jacobian_determinant[idx(i+1, j, k)] * u_comp[idx(i+1, j, k)] -
										   jacobian_determinant[idx(i-1, j, k)] * u_comp[idx(i-1, j, k)]) / (2 * dx_comp);
							
							TNum d_Jv_deta = (jacobian_determinant[idx(i, j+1, k)] * v_comp[idx(i, j+1, k)] -
											jacobian_determinant[idx(i, j-1, k)] * v_comp[idx(i, j-1, k)]) / (2 * dy_comp);
							
							TNum d_Jw_dzeta = (jacobian_determinant[idx(i, j, k+1)] * w_comp[idx(i, j, k+1)] -
											 jacobian_determinant[idx(i, j, k-1)] * w_comp[idx(i, j, k-1)]) / (2 * dz_comp);
							
							// Compute divergence
							div[index] = (d_Ju_dxi + d_Jv_deta + d_Jw_dzeta) / J;
						}
					}
				}
			}
			
			return div;
		}
		
		// Apply coordinate-specific corrections to the Navier-Stokes equations
		void applyCoordinateCorrections(VectorObj<TNum>& u_comp, VectorObj<TNum>& v_comp, 
									   VectorObj<TNum>& w_comp, TNum dt) {
			// For Cartesian coordinates, no corrections are needed
			if (mesh_type == MeshType::CARTESIAN) {
				return;
			}
			
			int n = nx * ny * nz;
			
			// For non-Cartesian coordinates, apply corrections based on Christoffel symbols
			for (int k = 1; k < nz - 1; ++k) {
				for (int j = 1; j < ny - 1; ++j) {
					for (int i = 1; i < nx - 1; ++i) {
						int index = idx(i, j, k);
						
						// Get velocity components at this point
						TNum u = u_comp[index];
						TNum v = v_comp[index];
						TNum w = w_comp[index];
						
						// Get Christoffel symbols at this point
						std::array<TNum, 27> symbols;
						
						if (mesh_type == MeshType::CURVILINEAR) {
							symbols = christoffel_symbols[index];
						} else {
							// Compute on-the-fly for simple coordinate systems
							symbols = computeChristoffelSymbols(i, j, k);
						}
						
						// Apply corrections based on the formula:
						// du^i/dt += -Γ^i_jk * u^j * u^k
						
						// Correction for u component (i=0)
						TNum u_correction = 0.0;
						for (int j = 0; j < 3; ++j) {
							for (int k = 0; k < 3; ++k) {
								TNum u_j = (j == 0) ? u : ((j == 1) ? v : w);
								TNum u_k = (k == 0) ? u : ((k == 1) ? v : w);
								u_correction -= symbols[0*9 + j*3 + k] * u_j * u_k;
							}
						}
						
						// Correction for v component (i=1)
						TNum v_correction = 0.0;
						for (int j = 0; j < 3; ++j) {
							for (int k = 0; k < 3; ++k) {
								TNum u_j = (j == 0) ? u : ((j == 1) ? v : w);
								TNum u_k = (k == 0) ? u : ((k == 1) ? v : w);
								v_correction -= symbols[1*9 + j*3 + k] * u_j * u_k;
							}
						}
						
						// Correction for w component (i=2)
						TNum w_correction = 0.0;
						for (int j = 0; j < 3; ++j) {
							for (int k = 0; k < 3; ++k) {
								TNum u_j = (j == 0) ? u : ((j == 1) ? v : w);
								TNum u_k = (k == 0) ? u : ((k == 1) ? v : w);
								w_correction -= symbols[2*9 + j*3 + k] * u_j * u_k;
							}
						}
						
						// Apply corrections
						u_comp[index] += dt * u_correction;
						v_comp[index] += dt * v_correction;
						w_comp[index] += dt * w_correction;
					}
				}
			}
		}
	};
	
	#endif // MESH_ADAPTER_HPP