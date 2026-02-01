//! # Well-Separated Pair Decomposition (WPSD)
//!
//! This crate provides an efficient implementation of the Well-Separated Pair Decomposition
//! data structure for geometric approximation algorithms.
//!
//! ## What is a WPSD
//!
//! A Well-Separated Pair Decomposition (WSPD) is a way to partition all pairs of points in
//! a d-dimensional point set into a smaller number of well-separated pairs. Two sets A and B are
//! **s-well separated** if they can be enclosed in balls of equal radius r such that
//! the distance between the balls is at least s * r.
//!
//! The key insight is that for a point set of size n, we can represent all $\theta(n^2)$ pairs using
//! only $O(s^d n)$ well-separated pairs, where d is the dimension and s is the separation factor.

mod bounding_box;
mod point;
mod split_tree;
mod wspd;

pub use point::{Point, Point2D, Point3D, Scalar, VecPoint};
pub use wspd::{WSPDBuilder, WSPDStats, WellSeparatedPair, WSPD};
