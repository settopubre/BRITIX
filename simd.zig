// simd.zig - SIMD operations for Britix (AVX2/AVX512)
// "Eight floats at once? I say, that's jolly fast!"

const std = @import("std");
const builtin = @import("builtin");

// Detect SIMD support at comptime
pub const have_avx2 = builtin.cpu.features.isEnabled(@field(std.Target.x86.Feature, "avx2"));
pub const have_avx512f = builtin.cpu.features.isEnabled(@field(std.Target.x86.Feature, "avx512f"));
pub const have_avx512vl = builtin.cpu.features.isEnabled(@field(std.Target.x86.Feature, "avx512vl"));

pub const Vector = @Vector(8, f32);  // 8 floats = 256 bits (AVX2)
pub const Vector512 = @Vector(16, f32);  // 16 floats = 512 bits (AVX512)

pub fn vec_add(a: Vector, b: Vector) Vector {
    return a + b;
}

pub fn vec_mul(a: Vector, b: Vector) Vector {
    return a * b;
}

pub fn vec_fma(a: Vector, b: Vector, c: Vector) Vector {
    // Fused multiply-add: a * b + c
    return a * b + c;
}

pub fn vec_exp(vec: Vector) Vector {
    // Use built-in exp for each element
    var result: Vector = undefined;
    for (0..8) |i| {
        result[i] = @exp(vec[i]);
    }
    return result;
}

pub fn matmul_simd(comptime M: usize, comptime N: usize, comptime K: usize, 
                   a: [*]const f32, b: [*]const f32, c: [*]f32) void {
    // Matrix multiplication C = A * B where:
    // A is M x K, B is K x N, C is M x N
    
    // Simplified scalar implementation for now
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

pub fn softmax_simd(logits: []f32) void {
    // Find max
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |val| {
        if (val > max_val) max_val = val;
    }
    
    // Compute exp and sum
    var sum: f32 = 0.0;
    for (0..logits.len) |i| {
        logits[i] = @exp(logits[i] - max_val);
        sum += logits[i];
    }
    
    // Normalize
    const inv_sum = 1.0 / sum;
    for (0..logits.len) |i| {
        logits[i] *= inv_sum;
    }
}

pub fn rms_norm_simd(x: []f32, weight: []const f32, eps: f32) void {
    // Compute RMS
    var sum_sq: f32 = 0.0;
    for (x) |val| {
        sum_sq += val * val;
    }
    
    const rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    
    // Apply norm and weight
    for (0..x.len) |i| {
        x[i] = x[i] * rms * weight[i];
    }
}

test "simd - vector addition" {
    const a: Vector = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b: Vector = .{ 8, 7, 6, 5, 4, 3, 2, 1 };
    const c = vec_add(a, b);
    
    try std.testing.expectEqual(@as(f32, 9.0), c[0]);
    try std.testing.expectEqual(@as(f32, 9.0), c[7]);
}

test "simd - matmul" {
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var c: [4]f32 = undefined;
    
    matmul_simd(2, 2, 2, &a, &b, &c);
    
    // Expected: [19, 22, 43, 50]
    try std.testing.expectApproxEqAbs(19.0, c[0], 0.001);
    try std.testing.expectApproxEqAbs(22.0, c[1], 0.001);
    try std.testing.expectApproxEqAbs(43.0, c[2], 0.001);
    try std.testing.expectApproxEqAbs(50.0, c[3], 0.001);
}
