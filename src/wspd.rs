use crate::bounding_box::are_well_separated;
use crate::point::Point;
use crate::split_tree::{SplitTree, SplitTreeNode};
use num_traits::{FromPrimitive, Zero};
use std::sync::Arc;

/// A well-separated pair in the decomposition.
///
/// Each pair consists of two nodes from the split tree, representing
/// two disjoint subsets of points that are s-well separated.
#[derive(Clone, Debug)]
pub struct WellSeparatedPair<P: Point> {
    /// First node in the pair.
    pub node_a: Arc<SplitTreeNode<P>>,
    /// Second node in the pair.
    pub node_b: Arc<SplitTreeNode<P>>,
}

impl<P: Point> WellSeparatedPair<P> {
    /// Returns the representative points from the first set (if available)
    pub fn representative_a(&self) -> Option<usize> {
        self.node_a.representative
    }
    /// Returns the representative points from the second set (if available)
    pub fn representative_b(&self) -> Option<usize> {
        self.node_b.representative
    }

    /// Returns all point indices in the first set.
    pub fn points_a(&self) -> Vec<usize> {
        SplitTree::collect_points(&self.node_a)
    }

    /// Returns all point indices in the second set.
    pub fn points_b(&self) -> Vec<usize> {
        SplitTree::collect_points(&self.node_b)
    }

    /// Returns the number of pairs represented by this well-separated pair.
    pub fn pair_count(&self) -> usize {
        self.node_a.size() * self.node_b.size()
    }
}

pub struct WSPD<P: Point> {
    /// The split tree
    tree: SplitTree<P>,
    /// The well-separated pairs
    pairs: Vec<WellSeparatedPair<P>>,
    /// The separation factor
    separation: P::Scalar,
}

impl<P: Point> WSPD<P> {
    /// Constructs a new WSPD with the given separation factor.
    ///
    /// # Arguments
    ///
    /// - `points`: The set of points to decompose.
    /// - `separation`: The separation factor s (larger values give fewer, better-separated pairs).
    ///
    /// # Time Complexity
    ///
    /// O(n log n + s^d n) where n is the number of points, d is the dimension, and s is the separation factor.
    ///
    /// # Panics
    ///
    /// Panics if the point set is empty or if separation is not positive
    pub fn new(points: Vec<P>, separation: P::Scalar) -> Self {
        assert!(!points.is_empty(), "Point set must not be empty");
        assert!(
            separation > P::Scalar::zero(),
            "Separation factor must be positive"
        );

        let tree = SplitTree::new(points);
        let pairs = Self::collect_pairs(&tree.root, separation);

        Self {
            tree,
            pairs,
            separation,
        }
    }

    /// Returns the number of well-separated pairs.
    pub fn num_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Returns a reference to the pairs.
    pub fn pairs(&self) -> &[WellSeparatedPair<P>] {
        &self.pairs
    }

    /// Returns a reference to the original points
    pub fn points(&self) -> &[P] {
        self.tree.points()
    }

    /// Returns the separation factor
    pub fn separation(&self) -> P::Scalar {
        self.separation
    }

    /// Returns the statistics about the WSPD
    pub fn stats(&self) -> WSPDStats {
        let total_point_pairs = self.pairs.iter().map(|p| p.pair_count()).sum();
        let n = self.points().len();
        let expected_pairs = n * (n - 1) / 2;

        WSPDStats {
            num_points: n,
            num_pairs: self.num_pairs(),
            total_point_pairs,
            expected_pairs,
            tree_nodes: self.tree.node_count(),
            tree_height: self.tree.height(),
        }
    }

    /// Computes well-separated pairs for a subtree.
    ///
    /// Recurses into each child independently to collect pairs *within* each
    /// child's subtree, then finds the cross pairs *between* the two
    /// children exactly once. Calling `find_well_separated(node.left,
    /// node.right, s)` from both sides (e.g. by starting from `(root,
    /// root)`) would visit every cross pair twice, once per traversal
    /// order, so each pair of siblings is only ever compared here.
    fn collect_pairs(node: &Arc<SplitTreeNode<P>>, s: P::Scalar) -> Vec<WellSeparatedPair<P>> {
        if node.is_leaf() {
            return Vec::new();
        }

        let mut pairs = Vec::new();
        if let Some(left) = &node.left {
            pairs.extend(Self::collect_pairs(left, s));
        }
        if let Some(right) = &node.right {
            pairs.extend(Self::collect_pairs(right, s));
        }
        if let (Some(left), Some(right)) = (&node.left, &node.right) {
            pairs.extend(Self::find_well_separated(left, right, s));
        }
        pairs
    }

    /// Finds well-separated pairs between two disjoint subtrees.
    fn find_well_separated(
        u: &Arc<SplitTreeNode<P>>,
        v: &Arc<SplitTreeNode<P>>,
        s: P::Scalar,
    ) -> Vec<WellSeparatedPair<P>> {
        let mut pairs = Vec::new();

        // check if u and v are well-separated
        if are_well_separated(&u.bbox, &v.bbox, s) {
            pairs.push(WellSeparatedPair {
                node_a: Arc::clone(u),
                node_b: Arc::clone(v),
            });
            return pairs;
        }

        // not well-separated, recurse on children
        // always split the node with larger level (smaller cell)
        // or the node with non-infinity level if one is a leaf
        let split_u = if v.is_leaf() {
            true
        } else if u.is_leaf() {
            false
        } else {
            u.level <= v.level
        };

        if split_u {
            // split u and recurse on its children with v
            if let Some(left) = &u.left {
                pairs.extend(Self::find_well_separated(left, v, s));
            }
            if let Some(right) = &u.right {
                pairs.extend(Self::find_well_separated(right, v, s));
            }
        } else {
            // split v and recurse on its children with u
            if let Some(left) = &v.left {
                pairs.extend(Self::find_well_separated(u, left, s));
            }
            if let Some(right) = &v.right {
                pairs.extend(Self::find_well_separated(u, right, s));
            }
        }

        pairs
    }

    pub fn all_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.pairs.iter().flat_map(|wsp| {
            let points_a = wsp.points_a();
            let points_b = wsp.points_b();
            points_a.into_iter().flat_map(move |a| {
                let points_b = points_b.clone(); // clone so each closure gets its own copy
                points_b.into_iter().filter(move |&b| b != a).map(move |b| {
                    if a < b {
                        (a, b)
                    } else {
                        (b, a)
                    }
                })
            })
        })
    }
}

/// Statistics about the WSPD
#[derive(Clone, Debug, Copy)]
pub struct WSPDStats {
    /// Number of points in the original set
    pub num_points: usize,
    /// Number of well-separated pairs
    pub num_pairs: usize,
    /// Total number of point pairs represented by the WSPD. Should always equal
    /// `expected_pairs`, since a correct WSPD covers every point pair exactly once.
    pub total_point_pairs: usize,
    /// Expected number of point pairs (n choose 2)
    pub expected_pairs: usize,
    /// Number of nodes in the split tree
    pub tree_nodes: usize,
    /// Height of the split tree
    pub tree_height: usize,
}

impl WSPDStats {
    /// Prints the statistics
    pub fn print(&self) {
        println!("WSPD Statistics:");
        println!("\tPoints: {}", self.num_points);
        println!("\tWell-Separated Pairs: {}", self.num_pairs);
        println!("\tPoint pairs covered: {}", self.total_point_pairs);
        println!("\tExpected point pairs: {}", self.expected_pairs);
        println!(
            "\tCompression Ratio: {:.2}x",
            self.expected_pairs as f64 / self.num_pairs as f64
        );
        println!("\tTree Nodes: {}", self.tree_nodes);
        println!("\tTree Height: {}", self.tree_height);
    }
}

/// Builder for constructing a WSPD with custom parameters.
pub struct WSPDBuilder<P: Point> {
    points: Vec<P>,
    separation: Option<P::Scalar>,
}

impl<P: Point> WSPDBuilder<P> {
    /// Creates a new WSPD builder with the given points.
    pub fn new(points: Vec<P>) -> Self {
        Self {
            points,
            separation: None,
        }
    }

    /// Sets the separation factor.
    ///
    /// Default is 2.0 if not specified.
    pub fn separation(mut self, separation: P::Scalar) -> Self {
        self.separation = Some(separation);
        self
    }

    /// Builds the WSPD with the specified parameters.
    pub fn build(self) -> WSPD<P> {
        let separation = self.separation.unwrap_or_else(|| {
            FromPrimitive::from_f64(2.0).expect("Failed to convert default separation")
        });
        WSPD::new(self.points, separation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point::Point2D;

    #[test]
    fn test_wspd_construction() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(10.0, 10.0),
        ];

        let wpsd = WSPD::new(points, 2.0);
        assert!(wpsd.num_pairs() > 0);

        let stats = wpsd.stats();
        assert_eq!(stats.num_points, 4);
        // WSPD may produce up to O(s^d * n) pairs, which can exceed n*(n-1)/2
        // for small point sets. The key property is that all_pairs covers all point pairs.
        assert!(stats.num_pairs > 0);
    }

    #[test]
    fn test_all_pairs_converge() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 2.0),
        ];
        let wpsd = WSPD::new(points, 2.0);
        let pair_set: std::collections::HashSet<(usize, usize)> = wpsd.all_pairs().collect();
        // should cover all unique pairs
        assert_eq!(pair_set.len(), 3); // C(3,2) = 3 pairs
    }

    #[test]
    fn test_no_duplicate_pairs() {
        // regression test: each point pair must be covered by exactly one
        // well-separated pair, not two (a prior bug counted every cross-subtree
        // pair twice, once per traversal order)
        for n in [5, 10, 25, 50] {
            let points: Vec<Point2D<f64>> = (0..n)
                .map(|i| Point2D::new(i as f64, (i * 7 % 13) as f64))
                .collect();
            let wpsd = WSPD::new(points, 2.0);
            let stats = wpsd.stats();

            assert_eq!(
                stats.total_point_pairs, stats.expected_pairs,
                "n={n}: total_point_pairs should equal n choose 2, found duplicate or missing pairs"
            );

            // all_pairs() without deduplication should also match exactly,
            // confirming no pair is emitted by more than one WellSeparatedPair
            assert_eq!(wpsd.all_pairs().count(), stats.expected_pairs);
        }
    }
}
