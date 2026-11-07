const std = @import("std");

pub const Layer = struct {
    weights: Matrix,
    biases: Matrix,
    last_input: ?Matrix = null,
    last_z: ?Matrix = null,
    grad_weights: ?Matrix = null,
    grad_biases: ?Matrix = null,

    pub fn forward(self: *Layer, input: Matrix) !Matrix {
        if (self.last_input) |m| m.deinit();
        if (self.last_z) |m| m.deinit();

        // Clone input values instead of sharing pointers
        const input_copy = try Matrix.empty_like(input);
        @memcpy(input_copy.values, input.values);
        self.last_input = input_copy;

        const z_linear = try input.mul(self.weights);
        defer z_linear.deinit();

        const z = try z_linear.addBias(self.biases);
        defer z.deinit();

        // Clone z values for the backward pass cache
        const z_copy = try Matrix.empty_like(z);
        @memcpy(z_copy.values, z.values);
        self.last_z = z_copy;

        return z.relu();
    }

    pub fn backward(self: *Layer, output_gradient: Matrix) !Matrix {
        const input = self.last_input orelse return error.NoForwardInputCache;
        const z = self.last_z orelse return error.NoForwardZCache;

        const z_deriv = try z.reluDerivative();
        defer z_deriv.deinit();

        const dZ = try output_gradient.mulElementWise(z_deriv);
        defer dZ.deinit();

        const input_t = try input.transpose();
        defer input_t.deinit();

        // Free old gradient caches if they exist
        if (self.grad_weights) |m| m.deinit();
        if (self.grad_biases) |m| m.deinit();

        // Assign to the layer's gradient fields so update() can see them
        self.grad_weights = try input_t.mul(dZ);
        self.grad_biases = try dZ.sumRows();

        const weights_t = try self.weights.transpose();
        defer weights_t.deinit();

        return dZ.mul(weights_t);
    }

    pub fn update(self: *Layer, learning_rate: f32) void {
        const dw = self.grad_weights orelse return;
        const db = self.grad_biases orelse return;
        for (0..self.weights.rows * self.weights.cols) |i| {
            self.weights.values[i] -= learning_rate * dw.values[i];
        }
        for (0..self.biases.rows * self.biases.cols) |i| {
            self.biases.values[i] -= learning_rate * db.values[i];
        }
    }

    pub fn deinit(self: *Layer) void {
        self.weights.deinit();
        self.biases.deinit();
        if (self.last_input) |m| m.deinit();
        if (self.last_z) |m| m.deinit();
        if (self.grad_weights) |m| m.deinit();
        if (self.grad_biases) |m| m.deinit();
    }
};

pub const Matrix = struct {
    allocator: std.mem.Allocator,
    values: []f32,
    rows: usize,
    cols: usize,

    pub fn init(allocator: std.mem.Allocator, values: []const f32, rows: usize, cols: usize) !Matrix {
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

    pub fn mulElementWise(self: Matrix, other: Matrix) !Matrix {
        std.debug.assert(self.rows == other.rows and self.cols == other.cols);
        var result = try Matrix.empty_like(self);
        for (0..self.rows * self.cols) |i| {
            result.values[i] = self.values[i] * other.values[i];
        }
        return result;
    }

    pub fn sumRows(self: Matrix) !Matrix {
        var result = try Matrix.empty(self.allocator, 1, self.cols);
        @memset(result.values, 0.0);
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                result.set(0, j, result.get(0, j) + self.get(i, j));
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

pub fn mseLoss(pred: Matrix, target: Matrix) !f32 {
    std.debug.assert(pred.rows == target.rows and pred.cols == target.cols);
    var sum: f32 = 0.0;
    const n: f32 = @floatFromInt(pred.rows * pred.cols);
    for (0..pred.rows * pred.cols) |i| {
        const diff = pred.values[i] - target.values[i];
        sum += diff * diff;
    }
    return sum / n;
}

pub fn mseDerivative(pred: Matrix, target: Matrix) !Matrix {
    std.debug.assert(pred.rows == target.rows and pred.cols == target.cols);
    var result = try Matrix.empty_like(pred);
    const n: f32 = @floatFromInt(pred.rows * pred.cols);
    for (0..pred.rows * pred.cols) |i| {
        result.values[i] = 2.0 * (pred.values[i] - target.values[i]) / n;
    }
    return result;
}

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
    const biases = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 1, 2 },
        1,
        2,
    );
    var layer = Layer{
        .weights = weights,
        .biases = biases,
    };
    defer layer.deinit(); // Moved to defer at the end

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

test "Training Loop" {
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    // Initialize random weights and biases
    const weights = try Matrix.random(std.testing.allocator, 2, 2, rand, .{ .min = -0.5, .max = 0.5 });
    const biases = try Matrix.random(std.testing.allocator, 1, 2, rand, .{ .min = -0.5, .max = 0.5 });

    var layer = Layer{
        .weights = weights,
        .biases = biases,
    };
    defer layer.deinit(); // Handles weights, biases, caches, and gradients safely at the end

    // Synthetic input data (batch of 2 samples, 2 features)
    const X = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 1.0, 0.5, 0.2, 1.0 },
        2,
        2,
    );
    defer X.deinit();

    // Target output data
    const Y = try Matrix.init(
        std.testing.allocator,
        &[_]f32{ 1.5, 2.0, 0.5, 1.2 },
        2,
        2,
    );
    defer Y.deinit();

    var initial_loss: f32 = 0.0;
    var final_loss: f32 = 0.0;
    const learning_rate: f32 = 0.01;

    for (0..100) |epoch| {
        const pred = try layer.forward(X);
        defer pred.deinit();

        const loss = try mseLoss(pred, Y);
        if (epoch == 0) initial_loss = loss;
        if (epoch == 99) final_loss = loss;

        const d_loss = try mseDerivative(pred, Y);
        defer d_loss.deinit();

        const dX = try layer.backward(d_loss);
        defer dX.deinit();

        layer.update(learning_rate);
    }

    // Verify that the loss decreased over training
    try std.testing.expect(final_loss < initial_loss);
}
