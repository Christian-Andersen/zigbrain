const std = @import("std");

pub const Layer = struct {
    weights: Matrix,
    biases: Matrix,

    pub fn forward(self: Layer, other: Matrix) !Matrix {
        const a = try other.mul(self.weights);
        defer a.deinit();
        const b = try a.addBias(self.biases);
        defer b.deinit();
        return b.relu();
    }
};

pub const Matrix = struct {
    allocator: std.mem.Allocator,
    values: []f32,
    rows: usize,
    cols: usize,

    pub fn init(allocator: std.mem.Allocator, values: []const f32, rows: usize, cols: usize) !Matrix {
        // Asset that the dimensions are logical
        std.debug.assert(values.len == rows * cols);
        const result = try Matrix.empty(allocator, rows, cols);
        @memcpy(result.values, values);
        return result;
    }

    pub fn deinit(self: Matrix) void {
        self.allocator.free(self.values);
    }

    pub fn empty(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const values = try allocator.alloc(f32, rows * cols);
        return .{
            .allocator = allocator,
            .values = values,
            .rows = rows,
            .cols = cols,
        };
    }

    pub fn empty_like(other: Matrix) !Matrix {
        return try Matrix.empty(
            other.allocator,
            other.rows,
            other.cols,
        );
    }

    pub fn random(allocator: std.mem.Allocator, rows: usize, cols: usize, rand: std.Random, args: struct {
        min: f32 = -1.0,
        max: f32 = 1.0,
    }) !Matrix {
        std.debug.assert(args.min <= args.max);
        var values = try allocator.alloc(f32, rows * cols);
        const difference = args.max - args.min;
        for (0..rows * cols) |i| {
            values[i] = args.min + (difference * rand.float(f32));
        }
        return .{
            .allocator = allocator,
            .values = values,
            .rows = rows,
            .cols = cols,
        };
    }

    pub fn getElementCount(self: Matrix) usize {
        return self.rows * self.cols;
    }

    pub fn get(self: Matrix, row: usize, col: usize) f32 {
        return self.values[row * self.cols + col];
    }

    pub fn set(self: Matrix, row: usize, col: usize, value: f32) void {
        self.values[row * self.cols + col] = value;
    }

    pub fn mul(self: Matrix, other: Matrix) !Matrix {
        std.debug.assert(self.cols == other.rows);
        var result = try Matrix.empty(self.allocator, self.rows, other.cols);
        for (0..result.rows) |i| {
            for (0..result.cols) |j| {
                var value: f32 = 0;
                for (0..self.cols) |k| {
                    value += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, value);
            }
        }
        return result;
    }

    pub fn add(self: Matrix, other: Matrix) !Matrix {
        std.debug.assert(self.rows == other.rows and self.cols == other.cols);
        var result = try Matrix.empty_like(self);
        for (0..result.rows) |i| {
            for (0..result.cols) |j| {
                result.set(i, j, self.get(i, j) + other.get(i, j));
            }
        }
        return result;
    }

    pub fn addBias(self: Matrix, bias: Matrix) !Matrix {
        std.debug.assert(bias.rows == 1 and self.cols == bias.cols);
        var result = try Matrix.empty_like(self);
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                const val = self.get(i, j) + bias.get(0, j);
                result.set(i, j, val);
            }
        }
        return result;
    }

    pub fn relu(self: Matrix) !Matrix {
        const result = try Matrix.empty_like(self);
        for (0..self.rows * self.cols) |i| {
            result.values[i] = @max(0.0, self.values[i]);
        }
        return result;
    }

    pub fn reluDerivative(self: Matrix) !Matrix {
        const result = try Matrix.empty_like(self);
        for (0..self.rows * self.cols) |i| {
            result.values[i] = if (self.values[i] > 0.0) 1.0 else 0.0;
        }
        return result;
    }

    pub fn transpose(self: Matrix) !Matrix {
        const result = try Matrix.empty(self.allocator, self.cols, self.rows);
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                result.set(j, i, self.get(i, j));
            }
        }
        return result;
    }

    pub fn isEqual(self: Matrix, other: Matrix) bool {
        if (self.rows != other.rows) {
            return false;
        }
        if (self.cols != other.cols) {
            return false;
        }
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                if (self.get(i, j) != other.get(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }
};

test "Matrix" {
    const matrix_a = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 1, 2, 3, 4, 5, 6 },
        2,
        3,
    );
    defer matrix_a.deinit();
    const matrix_b = try matrix_a.transpose();
    defer matrix_b.deinit();
    const expected_matrix_b = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 1, 4, 2, 5, 3, 6 },
        3,
        2,
    );
    defer expected_matrix_b.deinit();
    try std.testing.expect(matrix_b.isEqual(expected_matrix_b));
    const matrix_c = try matrix_b.transpose();
    defer matrix_c.deinit();
    try std.testing.expect(matrix_a.isEqual(matrix_c));
    try std.testing.expect(matrix_a.getElementCount() == 6);
    const result = try matrix_a.mul(matrix_b);
    defer result.deinit();
    const expected_result = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 14, 32, 32, 77 },
        2,
        2,
    );
    defer expected_result.deinit();
    try std.testing.expect(result.isEqual(expected_result));
    const matrix_d = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 1, 1, 1, 1, 1, 1 },
        2,
        3,
    );
    defer matrix_d.deinit();
    const added = try matrix_a.add(matrix_d);
    defer added.deinit();
    const expected_added = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 2, 3, 4, 5, 6, 7 },
        2,
        3,
    );
    defer expected_added.deinit();
    try std.testing.expect(added.isEqual(expected_added));
    var prng = std.Random.DefaultPrng.init(std.testing.random_seed);
    const rand = prng.random();
    const random_matrix = try Matrix.random(std.testing.allocator, 10, 10, rand, .{});
    defer random_matrix.deinit();
    for (0..100) |i| {
        try std.testing.expect(random_matrix.values[i] >= -1.0);
        try std.testing.expect(random_matrix.values[i] <= 1.0);
    }
    const relu_matrix = try random_matrix.relu();
    defer relu_matrix.deinit();
    for (0..100) |i| {
        try std.testing.expect(relu_matrix.values[i] >= 0.0);
    }
    const d_relu_matrix = try random_matrix.reluDerivative();
    defer d_relu_matrix.deinit();
    for (0..100) |i| {
        try std.testing.expect(d_relu_matrix.values[i] == 0.0 or d_relu_matrix.values[i] == 1.0);
    }
}

test "Layer" {
    const weights = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 1, 2, 3, 4, 5, 6 },
        3,
        2,
    );
    defer weights.deinit();
    const biases = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 1, 2 },
        1,
        2,
    );
    defer biases.deinit();
    const layer = Layer{
        .weights = weights,
        .biases = biases,
    };
    const other = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 1, 2, 3, 4, 5, 6 },
        2,
        3,
    );
    defer other.deinit();
    const result = try layer.forward(other);
    defer result.deinit();
    const expected_result = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 23, 30, 50, 66 },
        2,
        2,
    );
    defer expected_result.deinit();
    try std.testing.expect(result.isEqual(expected_result));
}
