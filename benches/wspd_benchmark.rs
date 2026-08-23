use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use wpsd::{Point, Point2D, SplitTree, WSPD};

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

fn bench_split_tree_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_tree_construction");
    for &n in &[100usize, 1_000, 10_000] {
        let points = random_points(n, 1);
        group.bench_with_input(BenchmarkId::from_parameter(n), &points, |b, points| {
            b.iter(|| SplitTree::new(black_box(points.clone())));
        });
    }
    group.finish();
}

fn bench_wspd_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("wspd_construction");
    for &n in &[100usize, 1_000, 10_000] {
        let points = random_points(n, 2);
        group.bench_with_input(BenchmarkId::from_parameter(n), &points, |b, points| {
            b.iter(|| WSPD::new(black_box(points.clone()), 2.0));
        });
    }
    group.finish();
}

fn bench_distance_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_sum");
    for &n in &[200usize, 1_000, 4_000] {
        let points = random_points(n, 3);

        group.bench_with_input(BenchmarkId::new("brute_force", n), &points, |b, points| {
            b.iter(|| brute_force_distance_sum(black_box(points)));
        });

        let wspd = WSPD::new(points.clone(), 2.0);
        group.bench_with_input(BenchmarkId::new("wspd_approx", n), &wspd, |b, wspd| {
            b.iter(|| wspd_approximate_distance_sum(black_box(wspd)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_split_tree_construction,
    bench_wspd_construction,
    bench_distance_sum
);
criterion_main!(benches);
