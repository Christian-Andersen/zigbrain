const std = @import("std");
const Matrix = @import("root.zig").Matrix;

fn load_images(io: std.Io, path: []const u8, allocator: std.mem.Allocator) ![]Matrix {
    const file = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(file);
    const num = std.mem.readInt(u32, file[4..8], .big);
    const rows = std.mem.readInt(u32, file[8..12], .big);
    const cols = std.mem.readInt(u32, file[12..16], .big);
    const size = rows * cols;
    std.debug.print("{any} - {any} - {any}\n", .{ num, rows, cols });
    const matrices = try allocator.alloc(Matrix, num);
    for (0..num) |i| {
        const start = 16 + (i * size);
        const end = start + size;
        matrices[i] = try Matrix.empty(
            allocator,
            rows,
            cols,
        );
        for (0..rows * cols, file[start..end]) |j, k| {
            matrices[i].values[j] = k;
        }
    }
    return matrices;
}

fn load_labels(io: std.Io, path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    const file = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    return file[8..file.len];
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const train_images = try load_images(io, "/tmp/mnist-data/train-images-idx3-ubyte", gpa);
    defer gpa.free(train_images);
    const test_images = try load_images(io, "/tmp/mnist-data/t10k-images-idx3-ubyte", gpa);
    defer gpa.free(test_images);
    const train_labels = try load_labels(io, "/tmp/mnist-data/train-labels-idx1-ubyte", gpa);
    defer gpa.free(train_labels);
    const test_labels = try load_labels(io, "/tmp/mnist-data/t10k-labels-idx1-ubyte", gpa);
    defer gpa.free(test_labels);
}
