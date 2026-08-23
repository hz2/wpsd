# wpsd

[![CI](https://github.com/hz2/wpsd/actions/workflows/ci.yml/badge.svg)](https://github.com/hz2/wpsd/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/wpsd.svg)](https://crates.io/crates/wpsd)
[![docs.rs](https://img.shields.io/docsrs/wpsd)](https://docs.rs/wpsd)
[![license](https://img.shields.io/crates/l/wpsd.svg)](LICENSE)

A Rust library for Well-Separated Pair Decomposition (WSPD) using split trees for d-dimensional point sets.

## Why WSPD?

A WSPD partitions all `n * (n - 1) / 2` pairs of points into a much smaller set of well-separated
*pairs of point sets*, each represented by a single pair of representative points. This turns
algorithms that need to touch every pair of points into ones that only need to touch `O(s^d * n)`
well-separated pairs, at the cost of a small, tunable approximation error. It's the basis for
techniques like Barnes-Hut style N-body approximation, approximate spanners, and approximate
closest-pair queries.

For example, approximating the sum of all pairwise distances in a point set
(`examples/approximate_distance_sum.rs`) with separation `s = 2.0`:

| n     | exact pairs | WSPD pairs | pair reduction | relative error |
| ----- | ----------- | ---------- | --------------- | --------------- |
| 100   | 4,950       | 663        | 7.5x             | 5.6%             |
| 1,000 | 499,500     | 8,793      | 56.8x            | 1.3%             |
| 5,000 | 12,497,500  | 50,689     | 246.6x           | 0.9%             |

Run it yourself with `cargo run --release --example approximate_distance_sum`.

## Features

- Generic `Point` trait supporting custom point types
- Built-in 2D (`Point2D`), 3D (`Point3D`), and N-dimensional (`VecPoint`) point implementations
- Axis-aligned bounding box calculations
- Split tree (compressed quadtree) construction
- Well-separated pair decomposition (`WSPD`, `WSPDBuilder`)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
wpsd = "0.1"
```

## Usage

### Basic Example

```rust
use wpsd::{Point2D, SplitTree};

fn main() {
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 1.0),
        Point2D::new(2.0, 0.5),
        Point2D::new(3.0, 2.0),
    ];

    let tree = SplitTree::new(points);
    println!("Tree root size: {}", tree.root.size());
}
```

### Using N-dimensional Points

```rust
use wpsd::{VecPoint, Point, SplitTree};

fn main() {
    let points = vec![
        VecPoint::new(vec![0.0, 0.0, 0.0]),
        VecPoint::new(vec![1.0, 1.0, 1.0]),
        VecPoint::new(vec![2.0, 0.5, 1.5]),
    ];

    // Compute distance between two points
    let dist = points[0].distance(&points[1]);
    println!("Distance: {}", dist);

    let tree = SplitTree::new(points);
}
```

### Custom Point Types

Implement the `Point` trait for your own types:

```rust
use wpsd::{Point, Scalar};

#[derive(Clone)]
struct MyPoint {
    coords: [f64; 3],
}

impl Point for MyPoint {
    type Scalar = f64;

    fn dim(&self) -> usize {
        3
    }

    fn coord(&self, dim: usize) -> Self::Scalar {
        self.coords[dim]
    }
}
```

## Benchmarks

Criterion benchmarks live in `benches/wspd_benchmark.rs` and cover split tree construction, WSPD
construction, and the approximate-distance-sum use case above compared against a brute-force
baseline. Run them with:

```sh
cargo bench
```

Representative results (uniform random 2D points, `s = 2.0`, measured on the author's machine —
run the benchmarks yourself for numbers on your hardware):

| Operation                   | n = 200/100 | n = 1,000 | n = 4,000/10,000 |
| ---------------------------- | ----------- | --------- | ------------------ |
| Split tree construction      | 35 µs       | 502 µs    | 5.65 ms             |
| WSPD construction (`s = 2.0`)| 133 µs      | 1.91 ms   | 23.95 ms            |
| Distance sum: brute force    | 31.7 µs     | 797 µs    | 12.69 ms            |
| Distance sum: WSPD-based     | 3.1 µs      | 17.2 µs   | 82.6 µs             |

The WSPD-based aggregate goes from roughly 10x faster than brute force at n = 200 to over 150x
faster at n = 4,000, matching the expected `O(n)`-ish vs `O(n^2)` scaling as the point set grows.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
