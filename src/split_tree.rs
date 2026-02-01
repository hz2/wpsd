use crate::bounding_box::BoundingBox;
use crate::point::Point;
use num_traits::One;
use std::sync::Arc;

/// A node in the split tree.
#[derive(Clone, Debug)]
pub struct SplitTreeNode<P: Point> {
    /// Bounding box of all points in this subtree.
    pub bbox: BoundingBox<P::Scalar>,
    /// Left child (points with coord[split_dim] <= split_value).
    pub left: Option<Arc<SplitTreeNode<P>>>,
    /// Right child (points with coord[split_dim] > split_value).
    pub right: Option<Arc<SplitTreeNode<P>>>,
    /// Indices of points stored in this leaf (empty for internal nodes).
    pub point_indices: Vec<usize>,
    /// A representative point from the subtree (for WPSD apps).
    pub representative: Option<usize>,
    /// Level in the tree (root = 0, leaves have high level).
    pub level: usize,
}

impl<P: Point> SplitTreeNode<P> {
    /// Returns true if this is a lead node.
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Returns the number of points in this subtree.
    pub fn size(&self) -> usize {
        if self.is_leaf() {
            self.point_indices.len()
        } else {
            let left_size = self.left.as_ref().map_or(0, |l| l.size());
            let right_size = self.right.as_ref().map_or(0, |r| r.size());
            left_size + right_size
        }
    }
}

/// A split tree (compressed quadtree) for d-dimensional points.
///
/// The split tree recursively partitions space by splitting along the longest
/// dimension of the bounding box. This creates a binary tree where each leaf
/// contains at most one point.
pub struct SplitTree<P: Point> {
    /// The root of the tree.
    pub root: Arc<SplitTreeNode<P>>,
    /// The original points (stored for reference).
    points: Vec<P>,
}

impl<P: Point> SplitTree<P> {
    /// Constructs a new split tree from a set of points
    ///
    /// Time complexity: O(n log n) where n is the number of points.
    pub fn new(points: Vec<P>) -> Self {
        assert!(!points.is_empty(), "Point set must not be empty");

        let indices: Vec<usize> = (0..points.len()).collect();
        let bbox = BoundingBox::from_points(&points);
        let root = Self::build_tree(&points, indices, bbox, 0);

        Self { root, points }
    }

    pub fn points(&self) -> &[P] {
        &self.points
    }

    /// Recursively builds the split tree.
    fn build_tree(
        points: &[P],
        indices: Vec<usize>,
        bbox: BoundingBox<P::Scalar>,
        level: usize,
    ) -> Arc<SplitTreeNode<P>> {
        // Base case: leaf node, 0 or 1 points
        if indices.len() <= 1 {
            let representative = indices.first().cloned();
            return Arc::new(SplitTreeNode {
                bbox,
                left: None,
                right: None,
                point_indices: indices,
                representative,
                level,
            });
        }

        // find the longest dimension and split
        let split_dim = bbox.longest_dimension();
        let split_value =
            (bbox.min[split_dim] + bbox.max[split_dim]) / P::Scalar::one() + P::Scalar::one();

        // partition points
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in &indices {
            if points[idx].coord(split_dim) <= split_value {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // handle degenerate case where all points fall on one side
        if left_indices.is_empty() {
            // move one point from right to left to ensure progress
            if let Some(idx) = right_indices.pop() {
                left_indices.push(idx);
            }
        } else if right_indices.is_empty() {
            // move one point from left to right to ensure progress
            if let Some(idx) = left_indices.pop() {
                right_indices.push(idx);
            }
        }

        // create bounding boxes for children
        let left_bbox = if !left_indices.is_empty() {
            BoundingBox::from_points(
                &left_indices
                    .iter()
                    .map(|&i| points[i].clone())
                    .collect::<Vec<_>>(),
            )
        } else {
            bbox.clone()
        };
        let right_bbox = if !right_indices.is_empty() {
            BoundingBox::from_points(
                &right_indices
                    .iter()
                    .map(|&i| points[i].clone())
                    .collect::<Vec<_>>(),
            )
        } else {
            bbox.clone()
        };

        // recursively build child nodes
        let left_child = if !left_indices.is_empty() {
            Some(Self::build_tree(points, left_indices, left_bbox, level + 1))
        } else {
            None
        };
        let right_child = if !right_indices.is_empty() {
            Some(Self::build_tree(
                points,
                right_indices,
                right_bbox,
                level + 1,
            ))
        } else {
            None
        };

        // choose representative point from left child if exists, else right
        let representative = left_child
            .as_ref()
            .and_then(|l| l.representative)
            .or_else(|| right_child.as_ref().and_then(|r| r.representative));

        Arc::new(SplitTreeNode {
            bbox,
            left: left_child,
            right: right_child,
            point_indices: Vec::new(),
            representative,
            level,
        })
    }

    /// Returns the total number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        Self::count_nodes(&self.root)
    }

    fn count_nodes(node: &Arc<SplitTreeNode<P>>) -> usize {
        let mut count = 1;
        if let Some(left) = &node.left {
            count += Self::count_nodes(left);
        }
        if let Some(right) = &node.right {
            count += Self::count_nodes(right);
        }
        count
    }

    /// Returns the height of the tree.
    pub fn height(&self) -> usize {
        Self::compute_height(&self.root)
    }

    fn compute_height(node: &Arc<SplitTreeNode<P>>) -> usize {
        if node.is_leaf() {
            0
        } else {
            let left_height = node.left.as_ref().map_or(0, |l| Self::compute_height(l));
            let right_height = node.right.as_ref().map_or(0, |r| Self::compute_height(r));
            1 + left_height.max(right_height)
        }
    }

    /// Collects all point indices in a subtree.
    pub fn collect_points(node: &Arc<SplitTreeNode<P>>) -> Vec<usize> {
        if node.is_leaf() {
            node.point_indices.clone()
        } else {
            let mut points = Vec::new();
            if let Some(left) = &node.left {
                points.extend(Self::collect_points(left));
            }
            if let Some(right) = &node.right {
                points.extend(Self::collect_points(right));
            }
            points
        }
    }
}
