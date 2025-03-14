#ifndef VTK_WRITER_HPP
#define VTK_WRITER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <memory>
#include <algorithm>
#include <type_traits>

/**
 * @brief Class for writing VTK files for visualization
 * 
 * This class provides functionality to write structured grid data to VTK files
 * that can be visualized in tools like ParaView or VisIt. It supports scalar,
 * vector, and tensor data fields.
 */
template <typename T>
class VTKWriter {
private:
    std::ofstream file;
    bool file_open = false;
    bool header_written = false;
    bool points_written = false;
    int num_points = 0;
    int num_cells = 0;
    
    // Helper function to write VTK header
    void writeHeader(const std::string& title) {
        file << "# vtk DataFile Version 3.0\n";
        file << title << "\n";
        file << "ASCII\n";
        header_written = true;
    }
    
    // Helper function to check if file is ready for writing data
    bool isReady() const {
        if (!file_open) {
            std::cerr << "Error: VTK file not open for writing" << std::endl;
            return false;
        }
        if (!header_written) {
            std::cerr << "Error: VTK header not written yet" << std::endl;
            return false;
        }
        if (!points_written) {
            std::cerr << "Error: Points not written yet" << std::endl;
            return false;
        }
        return true;
    }
    
    // Helper function to write point data header
    void writePointDataHeader() {
        file << "POINT_DATA " << num_points << "\n";
    }
    
    // Helper function to write cell data header
    void writeCellDataHeader() {
        file << "CELL_DATA " << num_cells << "\n";
    }
    
    // Helper function to check if a point data header has been written
    bool point_data_header_written = false;
    bool cell_data_header_written = false;
    
    // Helper function to ensure point data header is written
    void ensurePointDataHeader() {
        if (!point_data_header_written) {
            writePointDataHeader();
            point_data_header_written = true;
        }
    }
    
    // Helper function to ensure cell data header is written
    void ensureCellDataHeader() {
        if (!cell_data_header_written) {
            writeCellDataHeader();
            cell_data_header_written = true;
        }
    }

public:
    // Constructor
    VTKWriter() = default;
    
    // Destructor
    ~VTKWriter() {
        if (file_open) {
            file.close();
        }
    }
    
    /**
     * @brief Write structured grid to VTK file
     * 
     * @param filename The name of the VTK file to write
     * @param nx Number of points in x direction
     * @param ny Number of points in y direction
     * @param nz Number of points in z direction
     * @param x Array of x coordinates
     * @param y Array of y coordinates
     * @param z Array of z coordinates
     * @return true if successful, false otherwise
     */
    bool writeStructuredGrid(const std::string& filename, int nx, int ny, int nz,
                            const std::vector<T>& x, const std::vector<T>& y, const std::vector<T>& z) {
        // Open file
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
            return false;
        }
        file_open = true;
        
        // Write header
        writeHeader("Structured Grid");
        
        // Write dataset type
        file << "DATASET STRUCTURED_GRID\n";
        
        // Write dimensions
        file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
        
        // Calculate number of points
        num_points = nx * ny * nz;
        num_cells = (nx > 1 ? nx - 1 : 1) * (ny > 1 ? ny - 1 : 1) * (nz > 1 ? nz - 1 : 1);
        
        // Check if input arrays have correct size
        if (x.size() != num_points || y.size() != num_points || z.size() != num_points) {
            std::cerr << "Error: Input arrays size mismatch. Expected " << num_points 
                      << " points, got " << x.size() << ", " << y.size() << ", " << z.size() << std::endl;
            file.close();
            file_open = false;
            return false;
        }
        
        // Write points
        file << "POINTS " << num_points << " float\n";
        for (int i = 0; i < num_points; ++i) {
            file << std::setprecision(6) << x[i] << " " << y[i] << " " << z[i] << "\n";
        }
        
        points_written = true;
        point_data_header_written = false;
        cell_data_header_written = false;
        
        return true;
    }
    
    /**
     * @brief Write unstructured grid to VTK file
     * 
     * @param filename The name of the VTK file to write
     * @param points Vector of point coordinates (x1,y1,z1,x2,y2,z2,...)
     * @param cells Vector of cell connectivity (n,id1,id2,...,idn,n,id1,...)
     * @param cell_types Vector of cell types (VTK_VERTEX, VTK_LINE, etc.)
     * @return true if successful, false otherwise
     */
    bool writeUnstructuredGrid(const std::string& filename, 
                              const std::vector<T>& points,
                              const std::vector<int>& cells,
                              const std::vector<int>& cell_types) {
        // Open file
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
            return false;
        }
        file_open = true;
        
        // Write header
        writeHeader("Unstructured Grid");
        
        // Write dataset type
        file << "DATASET UNSTRUCTURED_GRID\n";
        
        // Write points
        num_points = points.size() / 3;
        file << "POINTS " << num_points << " float\n";
        for (int i = 0; i < num_points; ++i) {
            file << std::setprecision(6) 
                 << points[i*3] << " " 
                 << points[i*3+1] << " " 
                 << points[i*3+2] << "\n";
        }
        
        // Count total size of cell array
        int cell_array_size = 0;
        for (size_t i = 0; i < cells.size(); ) {
            int n = cells[i];
            cell_array_size += n + 1;
            i += n + 1;
        }
        
        // Write cells
        num_cells = cell_types.size();
        file << "CELLS " << num_cells << " " << cell_array_size << "\n";
        for (size_t i = 0; i < cells.size(); ) {
            int n = cells[i++];
            file << n;
            for (int j = 0; j < n; ++j) {
                file << " " << cells[i++];
            }
            file << "\n";
        }
        
        // Write cell types
        file << "CELL_TYPES " << num_cells << "\n";
        for (int type : cell_types) {
            file << type << "\n";
        }
        
        points_written = true;
        point_data_header_written = false;
        cell_data_header_written = false;
        
        return true;
    }
    
    /**
     * @brief Add scalar data to the VTK file
     * 
     * @param name Name of the scalar field
     * @param data Vector of scalar values
     * @param is_cell_data If true, data is associated with cells, otherwise with points
     * @return true if successful, false otherwise
     */
    bool addScalarData(const std::string& name, const std::vector<T>& data, bool is_cell_data = false) {
        if (!isReady()) return false;
        
        int expected_size = is_cell_data ? num_cells : num_points;
        
        if (data.size() != expected_size) {
            std::cerr << "Error: Data size mismatch. Expected " << expected_size 
                      << ", got " << data.size() << std::endl;
            return false;
        }
        
        if (is_cell_data) {
            ensureCellDataHeader();
        } else {
            ensurePointDataHeader();
        }
        
        file << "SCALARS " << name << " float 1\n";
        file << "LOOKUP_TABLE default\n";
        
        for (const auto& value : data) {
            file << std::setprecision(6) << value << "\n";
        }
        
        return true;
    }
    
    /**
     * @brief Add vector data to the VTK file
     * 
     * @param name Name of the vector field
     * @param x Vector of x components
     * @param y Vector of y components
     * @param z Vector of z components
     * @param is_cell_data If true, data is associated with cells, otherwise with points
     * @return true if successful, false otherwise
     */
    bool addVectorData(const std::string& name, 
                      const T* x, const T* y, const T* z,
                      bool is_cell_data = false) {
        if (!isReady()) return false;
        
        int expected_size = is_cell_data ? num_cells : num_points;
        
        if (is_cell_data) {
            ensureCellDataHeader();
        } else {
            ensurePointDataHeader();
        }
        
        file << "VECTORS " << name << " float\n";
        
        for (int i = 0; i < expected_size; ++i) {
            file << std::setprecision(6) << x[i] << " " << y[i] << " " << z[i] << "\n";
        }
        
        return true;
    }
    
    /**
     * @brief Add vector data to the VTK file
     * 
     * @param name Name of the vector field
     * @param data Vector of vector components (x1,y1,z1,x2,y2,z2,...)
     * @param is_cell_data If true, data is associated with cells, otherwise with points
     * @return true if successful, false otherwise
     */
    bool addVectorData(const std::string& name, 
                      const std::vector<T>& data,
                      bool is_cell_data = false) {
        if (!isReady()) return false;
        
        int expected_size = is_cell_data ? num_cells : num_points;
        
        if (data.size() != expected_size * 3) {
            std::cerr << "Error: Data size mismatch. Expected " << expected_size * 3
                      << ", got " << data.size() << std::endl;
            return false;
        }
        
        if (is_cell_data) {
            ensureCellDataHeader();
        } else {
            ensurePointDataHeader();
        }
        
        file << "VECTORS " << name << " float\n";
        
        for (int i = 0; i < expected_size; ++i) {
            file << std::setprecision(6) 
                 << data[i*3] << " " 
                 << data[i*3+1] << " " 
                 << data[i*3+2] << "\n";
        }
        
        return true;
    }
    
    /**
     * @brief Add tensor data to the VTK file
     * 
     * @param name Name of the tensor field
     * @param data Vector of tensor components (xx,xy,xz,yx,yy,yz,zx,zy,zz,...)
     * @param is_cell_data If true, data is associated with cells, otherwise with points
     * @return true if successful, false otherwise
     */
    bool addTensorData(const std::string& name, 
                      const std::vector<T>& data,
                      bool is_cell_data = false) {
        if (!isReady()) return false;
        
        int expected_size = is_cell_data ? num_cells : num_points;
        
        if (data.size() != expected_size * 9) {
            std::cerr << "Error: Data size mismatch. Expected " << expected_size * 9
                      << ", got " << data.size() << std::endl;
            return false;
        }
        
        if (is_cell_data) {
            ensureCellDataHeader();
        } else {
            ensurePointDataHeader();
        }
        
        file << "TENSORS " << name << " float\n";
        
        for (int i = 0; i < expected_size; ++i) {
            for (int j = 0; j < 3; ++j) {
                file << std::setprecision(6) 
                     << data[i*9+j*3] << " " 
                     << data[i*9+j*3+1] << " " 
                     << data[i*9+j*3+2] << "\n";
            }
            file << "\n";
        }
        
        return true;
    }
    
    /**
     * @brief Close the VTK file
     */
    void close() {
        if (file_open) {
            file.close();
            file_open = false;
            header_written = false;
            points_written = false;
            point_data_header_written = false;
            cell_data_header_written = false;
        }
    }
    
    /**
     * @brief Write a simple rectilinear grid to VTK file
     * 
     * @param filename The name of the VTK file to write
     * @param nx Number of points in x direction
     * @param ny Number of points in y direction
     * @param nz Number of points in z direction
     * @param dx Grid spacing in x direction
     * @param dy Grid spacing in y direction
     * @param dz Grid spacing in z direction
     * @param origin Origin of the grid (x0, y0, z0)
     * @return true if successful, false otherwise
     */
    bool writeRectilinearGrid(const std::string& filename, 
                             int nx, int ny, int nz,
                             T dx, T dy, T dz,
                             const std::array<T, 3>& origin = {0, 0, 0}) {
        // Generate coordinate arrays
        std::vector<T> x(nx), y(ny), z(nz);
        
        for (int i = 0; i < ny; ++i) y[i] = origin[1] + i * dy;
        for (int i = 0; i < nz; ++i) z[i] = origin[2] + i * dz;
        
        // Open file
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
            return false;
        }
        file_open = true;
        
        // Write header
        writeHeader("Rectilinear Grid");
        
        // Write dataset type
        file << "DATASET RECTILINEAR_GRID\n";
        
        // Write dimensions
        file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
        
        // Write coordinates
        file << "X_COORDINATES " << nx << " float\n";
        for (const auto& value : x) {
            file << std::setprecision(6) << value << "\n";
        }
        
        file << "Y_COORDINATES " << ny << " float\n";
        for (const auto& value : y) {
            file << std::setprecision(6) << value << "\n";
        }
        
        file << "Z_COORDINATES " << nz << " float\n";
        for (const auto& value : z) {
            file << std::setprecision(6) << value << "\n";
        }
        
        // Calculate number of points and cells
        num_points = nx * ny * nz;
        num_cells = (nx - 1) * (ny - 1) * (nz - 1);
        
        points_written = true;
        point_data_header_written = false;
        cell_data_header_written = false;
        
        return true;
    }
    
    /**
     * @brief Write a simple image data (uniform grid) to VTK file
     * 
     * @param filename The name of the VTK file to write
     * @param nx Number of points in x direction
     * @param ny Number of points in y direction
     * @param nz Number of points in z direction
     * @param dx Grid spacing in x direction
     * @param dy Grid spacing in y direction
     * @param dz Grid spacing in z direction
     * @param origin Origin of the grid (x0, y0, z0)
     * @return true if successful, false otherwise
     */
    bool writeImageData(const std::string& filename, 
                       int nx, int ny, int nz,
                       T dx, T dy, T dz,
                       const std::array<T, 3>& origin = {0, 0, 0}) {
        // Open file
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
            return false;
        }
        file_open = true;
        
        // Write header
        writeHeader("Image Data");
        
        // Write dataset type
        file << "DATASET STRUCTURED_POINTS\n";
        
        // Write dimensions
        file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
        
        // Write origin
        file << "ORIGIN " << origin[0] << " " << origin[1] << " " << origin[2] << "\n";
        
        // Write spacing
        file << "SPACING " << dx << " " << dy << " " << dz << "\n";
        
        // Calculate number of points and cells
        num_points = nx * ny * nz;
        num_cells = (nx - 1) * (ny - 1) * (nz - 1);
        
        points_written = true;
        point_data_header_written = false;
        cell_data_header_written = false;
        
        return true;
    }
    
    /**
     * @brief Write a polydata (points, lines, polygons) to VTK file
     * 
     * @param filename The name of the VTK file to write
     * @param points Vector of point coordinates (x1,y1,z1,x2,y2,z2,...)
     * @param vertices Optional vector of vertex indices
     * @param lines Optional vector of line indices (n,id1,id2,...,idn,n,id1,...)
     * @param polygons Optional vector of polygon indices (n,id1,id2,...,idn,n,id1,...)
     * @return true if successful, false otherwise
     */
    bool writePolyData(const std::string& filename, 
                      const std::vector<T>& points,
                      const std::vector<int>& vertices = {},
                      const std::vector<int>& lines = {},
                      const std::vector<int>& polygons = {}) {
        // Open file
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
            return false;
        }
        file_open = true;
        
        // Write header
        writeHeader("Poly Data");
        
        // Write dataset type
        file << "DATASET POLYDATA\n";
        
        // Write points
        num_points = points.size() / 3;
        file << "POINTS " << num_points << " float\n";
        for (int i = 0; i < num_points; ++i) {
            file << std::setprecision(6) 
                 << points[i*3] << " " 
                 << points[i*3+1] << " " 
                 << points[i*3+2] << "\n";
        }
        
        // Write vertices if provided
        if (!vertices.empty()) {
            int num_vertices = vertices.size();
            file << "VERTICES " << num_vertices << " " << num_vertices * 2 << "\n";
            for (int id : vertices) {
                file << "1 " << id << "\n";
            }
        }
        
        // Write lines if provided
        if (!lines.empty()) {
            // Count total size of line array
            int line_array_size = 0;
            for (size_t i = 0; i < lines.size(); ) {
                int n = lines[i];
                line_array_size += n + 1;
                i += n + 1;
            }
            
            int num_lines = line_array_size / (lines[0] + 1);
            file << "LINES " << num_lines << " " << line_array_size << "\n";
            for (size_t i = 0; i < lines.size(); ) {
                int n = lines[i++];
                file << n;
                for (int j = 0; j < n; ++j) {
                    file << " " << lines[i++];
                }
                file << "\n";
            }
        }
        
        // Write polygons if provided
        if (!polygons.empty()) {
            // Count total size of polygon array
            int polygon_array_size = 0;
            for (size_t i = 0; i < polygons.size(); ) {
                int n = polygons[i];
                polygon_array_size += n + 1;
                i += n + 1;
            }
            
            int num_polygons = polygon_array_size / (polygons[0] + 1);
            file << "POLYGONS " << num_polygons << " " << polygon_array_size << "\n";
            for (size_t i = 0; i < polygons.size(); ) {
                int n = polygons[i++];
                file << n;
                for (int j = 0; j < n; ++j) {
                    file << " " << polygons[i++];
                }
                file << "\n";
            }
        }
        
        num_cells = 0;
        if (!vertices.empty()) num_cells += vertices.size();
        if (!lines.empty()) {
            for (size_t i = 0; i < lines.size(); ) {
                i += lines[i] + 1;
                num_cells++;
            }
        }
        if (!polygons.empty()) {
            for (size_t i = 0; i < polygons.size(); ) {
                i += polygons[i] + 1;
                num_cells++;
            }
        }
        
        points_written = true;
        point_data_header_written = false;
        cell_data_header_written = false;
        
        return true;
    }
};

#endif // VTK_WRITER_HPP