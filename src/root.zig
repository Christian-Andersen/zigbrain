const std = @import("std");

pub const Matrix = struct {
    values: []f32,
    rows: usize,
    columns: usize,

    pub fn getElementCount(self: Matrix) usize {
        return self.rows * self.columns;
    }
};

test "Matrix" {
    var values = [_]f32{ 1, 2, 3, 4 };
    const matrix = Matrix{
        .values = &values,
        .rows = 2,
        .columns = 2,
    };
    try std.testing.expect(matrix.getElementCount() == 4);
}
