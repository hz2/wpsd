use crate::point::{Point, Scalar};

/// An axis-aligned bounding box in a d-dimensional space.
///
/// Represented as intervals [min, max] for each dimension.
#[derive(Clone, Debug)]
pub struct BoundingBox<T: Scalar> {
    /// Minimum coordinates for each dimension.
    pub min: Vec<T>,
    /// Maximum coordinates for each dimension.
    pub max: Vec<T>,
}

impl<T: Scalar> BoundingBox<T> {
    /// Creates a new bounding box given min and max coordinates.
    pub fn new(min: Vec<T>, max: Vec<T>) -> Self {
        assert_eq!(
            min.len(),
            max.len(),
            "Min and max must have the same dimension"
        );
        assert!(
            min.iter().zip(&max).all(|(a, b)| a <= b),
            "Each min coordinate must be less than or equal to the corresponding max coordinate"
        );
        Self { min, max }
    }

    /// Creates a bounding box from a set of points.
    pub fn from_points<P: Point<Scalar = T>>(points: &[P]) -> Self {
        assert!(!points.is_empty(), "Point set must not be empty");
        let dim = points[0].dim();
        let mut min = vec![T::infinity(); dim];
        let mut max = vec![T::neg_infinity(); dim];

        for point in points {
            for d in 0..dim {
                let coord = point.coord(d);
                if coord < min[d] {
                    min[d] = coord;
                }
                if coord > max[d] {
                    max[d] = coord;
                }
            }
        }
        Self { min, max }
    }

    /// Returns the dimension of the bounding box.
    pub fn dim(&self) -> usize {
        self.min.len()
    }

    /// Returns the length of the longest side
    pub fn max_side_length(&self) -> T {
        (0..self.dim())
            .map(|d| self.max[d] - self.min[d])
            .fold(T::neg_infinity(), |acc, x| if x > acc { x } else { acc })
    }

    /// Returns the side length for a specific dimension.
    pub fn side_length(&self, dim: usize) -> T {
        self.max[dim] - self.min[dim]
    }

    /// Returns the dimension with the longest side.
    pub fn longest_dimension(&self) -> usize {
        (0..self.dim())
            .max_by(|&a, &b| {
                let len_a = self.side_length(a);
                let len_b = self.side_length(b);
                len_a.partial_cmp(&len_b).unwrap()
            })
            .unwrap()
    }

    /// Returns the center point of the bounding box.
    pub fn center(&self) -> Vec<T> {
        (0..self.dim())
            .map(|d| (self.min[d] + self.max[d]) / (T::one() + T::one()))
            .collect()
    }

    /// Computes the radius of the smallest ball enclosing this box.
    ///
    /// This is half the diagonal length of the bounding box.
    pub fn enclosing_radius(&self) -> T {
        let diagonal_sq: T = { 0..self.dim() }
            .map(|d| {
                let len = self.max[d] - self.min[d];
                len * len
            })
            .fold(T::zero(), |acc, x| acc + x);
        diagonal_sq.sqrt() / (T::one() + T::one())
    }

    /// Returns the center of the smallest ball enclosing this box.
    pub fn enclosing_center(&self) -> Vec<T> {
        self.center()
    }

    /// Computes the minimum distance between this box and another box.
    pub fn min_distance(&self, other: &Self) -> T {
        assert_eq!(
            self.dim(),
            other.dim(),
            "Bounding boxes must have the same dimension"
        );
        let dist_sq: T = (0..self.dim())
            .map(|d| {
                if self.max[d] < other.min[d] {
                    let diff = other.min[d] - self.max[d];
                    diff * diff
                } else if other.max[d] < self.min[d] {
                    let diff = self.min[d] - other.max[d];
                    diff * diff
                } else {
                    T::zero()
                }
            })
            .fold(T::zero(), |acc, x| acc + x);
        dist_sq.sqrt()
    }

    /// Checks if a point is inside the bounding box.
    pub fn contains<P: Point<Scalar = T>>(&self, point: &P) -> bool {
        (0..self.dim()).all(|d| {
            let coord = point.coord(d);
            coord >= self.min[d] && coord <= self.max[d]
        })
    }

    /// Splits the bounding box along the given dimension at the given value.
    pub fn split(&self, dim: usize, value: T) -> (Self, Self) {
        let mut left_max = self.max.clone();
        left_max[dim] = value;
        let mut right_min = self.min.clone();
        right_min[dim] = value;

        (
            BoundingBox::new(self.min.clone(), left_max),
            BoundingBox::new(right_min, self.max.clone()),
        )
    }

    /// Splits the bounding box along its longest dimension at the midpoint.
    pub fn split_longest(&self) -> (Self, Self, usize) {
        let dim = self.longest_dimension();
        let mid = (self.min[dim] + self.max[dim]) / (T::one() + T::one());
        let (left, right) = self.split(dim, mid);
        (left, right, dim)
    }
}

/// Checks if two bounding boxes are well-separated
///
/// Two boxes are s-well separated if they can be enclosed in balls of radius r
/// such that the distance between the balls is at least s * r.
pub fn are_well_separated<T: Scalar>(box1: &BoundingBox<T>, box2: &BoundingBox<T>, s: T) -> bool {
    let r1 = box1.enclosing_radius();
    let r2 = box2.enclosing_radius();
    let r = r1.max(r2);

    if r.is_zero() {
        // both points (leaves), always well-separated
        return true;
    }

    let c1 = box1.enclosing_center();
    let c2 = box2.enclosing_center();

    // distance between centers
    let center_dist_sq = c1
        .iter()
        .zip(&c2)
        .map(|(a, b)| {
            let diff = *a - *b;
            diff * diff
        })
        .fold(T::zero(), |acc, x| acc + x);
    let center_dist = center_dist_sq.sqrt();

    // distance between ball surfaces
    let surface_dist = center_dist - (r1 + r2);
    surface_dist >= s * r
}
