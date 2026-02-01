use num_traits::{Float, FromPrimitive, Zero};
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

/// Trait for numeric types used in geometric computations.
pub trait Scalar: Float + FromPrimitive + Send + Sync + Debug + 'static {}

// Blanket implementations for common float types
impl Scalar for f32 {}
impl Scalar for f64 {}

/// A point in d-dimensional euclidean space.
///
/// This trait allows the WPSD to work with various point representations.
/// Implement this trait for your custom point types.
pub trait Point: Clone + Send + Sync {
    /// The scalar type (e.g. f32, f64) used for the coordinates.
    type Scalar: Scalar;
    /// Returns the dimensionality of the point.
    fn dim(&self) -> usize;
    /// Returns the coordinate at the given dimension.
    fn coord(&self, dim: usize) -> Self::Scalar;
    /// Computes the squared Euclidean distance to another point.
    fn distance_squared(&self, other: &Self) -> Self::Scalar {
        (0..self.dim())
            .map(|i| {
                let diff = self.coord(i) - other.coord(i);
                diff * diff
            })
            .fold(Self::Scalar::zero(), |acc, x| acc + x)
    }
    /// Computes the Euclidean distance to another point.
    fn distance(&self, other: &Self) -> Self::Scalar {
        self.distance_squared(other).sqrt()
    }
}

/// A simple d-dimensional point implementation using a vector.
#[derive(Clone, Debug, PartialEq)]
pub struct VecPoint<T: Scalar> {
    coords: Vec<T>,
}

impl<T: Scalar> VecPoint<T> {
    /// Creates a new point from a vector of coordinates.
    pub fn new(coords: Vec<T>) -> Self {
        assert!(!coords.is_empty(), "Point must have at least one dimension");
        Self { coords }
    }

    /// Creates a new point from a slice of coordinates.
    pub fn from_slice(coords: &[T]) -> Self {
        Self::new(coords.to_vec())
    }

    /// Returns a reference to the underlying coordinates.
    pub fn as_slice(&self) -> &[T] {
        &self.coords
    }
}

impl<T: Scalar> Point for VecPoint<T> {
    type Scalar = T;

    fn dim(&self) -> usize {
        self.coords.len()
    }

    fn coord(&self, dim: usize) -> Self::Scalar {
        self.coords[dim]
    }
}

impl<T: Scalar> Index<usize> for VecPoint<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl<T: Scalar> IndexMut<usize> for VecPoint<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

/// A 2D point with explicit x and y coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2D<T: Scalar> {
    pub x: T,
    pub y: T,
}

impl<T: Scalar> Point2D<T> {
    /// Creates a new 2D point.
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T: Scalar> Point for Point2D<T> {
    type Scalar = T;

    fn dim(&self) -> usize {
        2
    }

    fn coord(&self, dim: usize) -> Self::Scalar {
        match dim {
            0 => self.x,
            1 => self.y,
            _ => panic!("Dimension out of bounds for Point2D"),
        }
    }
}

/// A 3D point with explicit x, y, and z coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3D<T: Scalar> {
    pub x: T,
    pub y: T,
    pub z: T,
}
impl<T: Scalar> Point3D<T> {
    /// Creates a new 3D point.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T: Scalar> Point for Point3D<T> {
    type Scalar = T;

    fn dim(&self) -> usize {
        3
    }

    fn coord(&self, dim: usize) -> Self::Scalar {
        match dim {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("Dimension out of bounds for Point3D"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_point() {
        let p1 = VecPoint::new(vec![1.0f64, 2.0, 3.0]);
        let p2 = VecPoint::new(vec![4.0f64, 5.0, 6.0]);

        assert_eq!(p1.dim(), 3);
        assert_eq!(p1.coord(0), 1.0);
        assert_eq!(p1.distance_squared(&p2), 27.0);
        assert_eq!(p1.distance(&p2), 5.196152422706632);
    }

    #[test]
    fn test_point2d() {
        let p1 = Point2D::new(1.0f32, 2.0);
        let p2 = Point2D::new(4.0f32, 6.0);

        assert_eq!(p1.dim(), 2);
        assert_eq!(p1.coord(0), 1.0);
        assert_eq!(p1.distance_squared(&p2), 25.0);
        assert_eq!(p1.distance(&p2), 5.0);
    }

    #[test]
    fn test_point3d() {
        let p1 = Point3D::new(1.0f64, 2.0, 3.0);
        let p2 = Point3D::new(4.0f64, 6.0, 8.0);
        assert_eq!(p1.dim(), 3);
        assert_eq!(p1.coord(1), 2.0);
        assert_eq!(p1.distance_squared(&p2), 50.0);
        assert_eq!(p1.distance(&p2), 7.0710678118654755);
    }
}
