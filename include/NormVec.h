#ifndef NORM_VEC_H
#define NORM_VEC_H

#include <cmath>

/**
 * @brief Represents a normalized 3D vector.
 *
 * Typically used to represent surface normals.
 */
struct NormVec {
    double x;   ///< X component of the vector.
    double y;   ///< Y component of the vector.
    double z;   ///< Z component of the vector.
    double len; ///< Length of the vector before normalization.

    /**
     * @brief Default constructor. Initializes vector to zero.
     */
    NormVec() : x(0), y(0), z(0), len(0) {}

    /**
     * @brief Constructs a normal vector from a triangle's vertices.
     *
     * Calculates the normal vector of the triangle defined by vertices
     * starting at the given offset in the wall array. The vector is normalized.
     *
     * @param wall Pointer to the array containing vertex coordinates.
     * @param offset Offset in the wall array where the triangle's vertices start.
     */
    inline NormVec(const double* wall, int offset) {
        // 1. Point directly to the data. 
        // This creates no new arrays, just looks at existing memory.
        const double* p0 = &wall[offset];     // or wall + offset
        const double* p1 = &wall[offset + 3];
        const double* p2 = &wall[offset + 6];

        // 2. Calculate vector components using scalars (doubles).
        // Compilers will map these directly to CPU registers (XMM/YMM), 
        // avoiding memory writes entirely.
        double ax = p1[0] - p0[0];
        double ay = p1[1] - p0[1];
        double az = p1[2] - p0[2];

        double bx = p2[0] - p0[0];
        double by = p2[1] - p0[1];
        double bz = p2[2] - p0[2];

        // 3. Cross product
        x = ay * bz - az * by;
        y = az * bx - ax * bz;
        z = ax * by - ay * bx;

        // 4. Normalize
        len = std::sqrt(x * x + y * y + z * z);
        
        // Prevent division by zero if points are identical/collinear
        if (len > 0) {
            double invLen = 1.0 / len; // Multiplication is faster than division
            x *= invLen;
            y *= invLen;
            z *= invLen;
        }
    }
};

#endif // NORM_VEC_H