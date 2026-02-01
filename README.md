# wpsd

A Rust library for Well-Separated Pair Decomposition (WSPD) using split trees for d-dimensional point sets.

## Features

- Generic `Point` trait supporting custom point types
- Built-in 2D (`Point2D`) and N-dimensional (`VecPoint`) point implementations
- Axis-aligned bounding box calculations
- Split tree (compressed quadtree) construction

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
