//! Demonstrates the practical payoff of a WSPD: approximating the sum of all
//! pairwise distances in a point set using far fewer operations than the
//! brute-force O(n^2) approach.
//!
//! For each well-separated pair, every point in set A is within a bounded
//! relative error of every point in set B (controlled by the separation
//! factor). So instead of summing n*(n-1)/2 exact distances, we sum
//! `pair_count()` copies of a single representative-to-representative
//! distance per well-separated pair, of which there are only O(s^d * n).
//!
//! This kind of approximate all-pairs aggregate is the basis of techniques
//! like Barnes-Hut n-body approximation and approximate spanners.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use wpsd::{Point, Point2D, WSPD};

fn random_points(n: usize, seed: u64) -> Vec<Point2D<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| Point2D::new(rng.random_range(0.0..1000.0), rng.random_range(0.0..1000.0)))
        .collect()
}

fn brute_force_distance_sum(points: &[Point2D<f64>]) -> f64 {
    let mut sum = 0.0;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            sum += points[i].distance(&points[j]);
        }
    }
    sum
}

fn wspd_approximate_distance_sum(wspd: &WSPD<Point2D<f64>>) -> f64 {
    let points = wspd.points();
    wspd.pairs()
        .iter()
        .map(|pair| {
            let a = pair.representative_a().map(|i| &points[i]);
            let b = pair.representative_b().map(|i| &points[i]);
            match (a, b) {
                (Some(a), Some(b)) => a.distance(b) * pair.pair_count() as f64,
                _ => 0.0,
            }
        })
        .sum()
}

fn main() {
    println!("n\texact pairs\twspd pairs\treduction\texact sum\tapprox sum\trel. error");

    for &n in &[100, 1_000, 5_000] {
        let points = random_points(n, 42);
        let exact_pairs = n * (n - 1) / 2;

        let exact_sum = brute_force_distance_sum(&points);

        let wspd = WSPD::new(points, 2.0);
        let approx_sum = wspd_approximate_distance_sum(&wspd);
        let rel_error = (approx_sum - exact_sum).abs() / exact_sum;

        println!(
            "{}\t{}\t{}\t{:.1}x\t{:.1}\t{:.1}\t{:.4}%",
            n,
            exact_pairs,
            wspd.num_pairs(),
            exact_pairs as f64 / wspd.num_pairs() as f64,
            exact_sum,
            approx_sum,
            rel_error * 100.0
        );
    }
}
