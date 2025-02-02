// fmm_gtest.cpp

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>

namespace FASTSolver {

// Fast Multipole Method implementation for 2D Laplace (logarithmic kernel)
template<typename T = double>
class FMM {
public:
    // Parameters for the FMM
    static constexpr int MAX_PARTICLES = 10;  // maximum particles per leaf cell
    static constexpr int P = 3;               // expansion order (using terms 0..P)
    static constexpr double THETA = 0.5;        // opening angle criterion

    // Particle structure
    struct Particle {
        T x, y, q;  // position (x,y) and charge q
        Particle(T x_, T y_, T q_) : x(x_), y(y_), q(q_) {}
    };

    // Cell structure for quad-tree
    struct Cell {
        T x_min, x_max, y_min, y_max;  // Bounding box
        T cx, cy;                    // Center coordinates
        std::vector<Particle*> particles;
        std::vector<Cell*> children;
        std::vector<std::complex<T>> multipole;
        bool is_leaf;

        Cell(T xmin, T xmax, T ymin, T ymax)
            : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax), is_leaf(true)
        {
            cx = static_cast<T>(0.5) * (xmin + xmax);
            cy = static_cast<T>(0.5) * (ymin + ymax);
            multipole.resize(P + 1, 0.0);
        }

        ~Cell() {
            // Recursively delete children cells
            for (auto child : children) {
                delete child;
            }
        }
    };

    // Build quad-tree recursively. Returns pointer to the (possibly subdivided) cell.
    Cell* buildTree(Cell* cell) {
        if (cell->particles.size() <= MAX_PARTICLES)
            return cell;

        T midx = cell->cx;
        T midy = cell->cy;

        // Create four children (quadrants)
        Cell* child1 = new Cell(cell->x_min, midx, cell->y_min, midy);
        Cell* child2 = new Cell(midx, cell->x_max, cell->y_min, midy);
        Cell* child3 = new Cell(cell->x_min, midx, midy, cell->y_max);
        Cell* child4 = new Cell(midx, cell->x_max, midy, cell->y_max);

        // Distribute particles into the appropriate quadrant
        for (Particle* p : cell->particles) {
            if (p->x <= midx) {
                if (p->y <= midy)
                    child1->particles.push_back(p);
                else
                    child3->particles.push_back(p);
            } else {
                if (p->y <= midy)
                    child2->particles.push_back(p);
                else
                    child4->particles.push_back(p);
            }
        }

        // Clear parent's particle list and mark as non-leaf
        cell->particles.clear();
        cell->is_leaf = false;
        cell->children = {child1, child2, child3, child4};

        // Recursively build the tree for children that have particles
        for (Cell* child : cell->children) {
            if (!child->particles.empty())
                buildTree(child);
        }

        return cell;
    }

    // Compute multipole expansion for leaf cells directly from particle data.
    void computeLeafMultipole(Cell* cell) {
        std::complex<T> zc(cell->cx, cell->cy);
        for (Particle* p : cell->particles) {
            std::complex<T> z(p->x, p->y);
            std::complex<T> dz = z - zc;
            std::complex<T> term(1.0, 0.0);
            for (int n = 0; n <= P; n++) {
                cell->multipole[n] += p->q * term;
                term *= dz;
            }
        }
    }

    // Helper function for computing the binomial coefficient "n choose k"
    T binomialCoeff(int n, int k) {
        T res = 1.0;
        for (int i = 1; i <= k; i++) {
            res *= static_cast<T>(n - i + 1) / i;
        }
        return res;
    }

    // Upward pass: combine child multipole expansions into the parent's multipole expansion.
    void upwardPass(Cell* cell) {
        if (cell->is_leaf) {
            computeLeafMultipole(cell);
            return;
        }

        // First, compute multipole expansions for the children.
        for (Cell* child : cell->children)
            upwardPass(child);

        // Reset parent's multipole expansion.
        std::fill(cell->multipole.begin(), cell->multipole.end(), 0.0);

        std::complex<T> z_parent(cell->cx, cell->cy);
        for (Cell* child : cell->children) {
            // Skip empty child cells.
            if (child->particles.empty() && child->children.empty())
                continue;

            std::complex<T> z_child(child->cx, child->cy);
            std::complex<T> d = z_child - z_parent;

            // Translate the child's multipole expansion to the parent's center.
            for (int n = 0; n <= P; n++) {
                std::complex<T> sum = 0.0;
                for (int k = 0; k <= n; k++) {
                    sum += binomialCoeff(n, k) * child->multipole[k] * std::pow(d, n - k);
                }
                cell->multipole[n] += sum;
            }
        }
    }

    // Check if a cell is well-separated from target point z using the opening angle criterion.
    bool wellSeparated(Cell* cell, const std::complex<T>& z) {
        T s = std::max(cell->x_max - cell->x_min, cell->y_max - cell->y_min);
        std::complex<T> zc(cell->cx, cell->cy);
        T d = std::abs(z - zc);
        return (s / d < THETA);
    }

    // Evaluate the potential at target point z.
    // For well-separated cells, use the multipole expansion; otherwise use direct summation.
    T evaluatePotential(Cell* cell, const std::complex<T>& z) {
        T potential = 0.0;
        if (wellSeparated(cell, z)) {
            std::complex<T> zc(cell->cx, cell->cy);
            std::complex<T> dz = z - zc;
            T r = std::abs(dz);
            if (r == 0) return 0.0; // Avoid division by zero
            std::complex<T> sum = 0.0;
            // Monopole term (n=0) uses the logarithmic kernel.
            sum += cell->multipole[0] * std::log(dz);
            for (int n = 1; n <= P; n++) {
                sum += cell->multipole[n] / (static_cast<T>(n) * std::pow(dz, n));
            }
            potential = -std::real(sum) / (2 * M_PI);
        } else if (cell->is_leaf) {
            for (Particle* p : cell->particles) {
                T dx = z.real() - p->x;
                T dy = z.imag() - p->y;
                T r = std::sqrt(dx * dx + dy * dy);
                if (r != 0)
                    potential += -p->q * std::log(r);
            }
            potential /= (2 * M_PI);
        } else {
            for (Cell* child : cell->children) {
                potential += evaluatePotential(child, z);
            }
        }
        return potential;
    }
};

} // namespace FASTSolver

