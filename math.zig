// math.zig - Fast math functions for Britix
// "Mathematics, but make it snappy! What what!"

const std = @import("std");
const builtin = @import("builtin");

pub fn exp(x: f32) f32 {
    // Just use Zig's built-in exp - it's fast and correct
    return @exp(x);
}

pub fn exp_simd(x: @Vector(8, f32)) @Vector(8, f32) {
    // SIMD exp using built-in
    return @exp(x);
}

pub fn erf(x: f32) f32 {
    // Error function approximation (max error 1.5e-7)
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    
    const sign = if (x < 0) -1.0 else 1.0;
    const abs_x = @abs(x);
    
    const t = 1.0 / (1.0 + p * abs_x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-abs_x * abs_x);
    
    return sign * y;
}

pub fn softmax(logits: []f32) void {
    // Find max for numerical stability
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |val| {
        if (val > max_val) max_val = val;
    }
    
    // Compute exponentials and sum
    var sum: f32 = 0.0;
    for (logits) |*val| {
        val.* = @exp(val.* - max_val);
        sum += val.*;
    }
    
    // Normalize
    const inv_sum = 1.0 / sum;
    for (logits) |*val| {
        val.* *= inv_sum;
    }
}

pub fn softmax_simd(logits: []f32) void {
    const vec_size = 8;
    var i: usize = 0;
    
    // Find max using scalar (SIMD max is tricky)
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |val| {
        if (val > max_val) max_val = val;
    }
    
    // Compute exponentials and sum
    i = 0;
    var sum: f32 = 0.0;
    const max_splat = @as(@Vector(8, f32), @splat(max_val));
    
    while (i + vec_size <= logits.len) : (i += vec_size) {
        const vec = logits[i..][0..vec_size].*;
        const exp_vec = @exp(vec - max_splat);
        logits[i..][0..vec_size].* = exp_vec;
        
        // Horizontal sum
        var lane_sum: f32 = 0.0;
        for (0..vec_size) |j| {
            lane_sum += exp_vec[j];
        }
        sum += lane_sum;
    }
    
    // Handle remaining
    for (i..logits.len) |j| {
        logits[j] = @exp(logits[j] - max_val);
        sum += logits[j];
    }
    
    // Normalize
    const inv_sum = 1.0 / sum;
    i = 0;
    const inv_sum_splat = @as(@Vector(8, f32), @splat(inv_sum));
    
    while (i + vec_size <= logits.len) : (i += vec_size) {
        const vec = logits[i..][0..vec_size].*;
        logits[i..][0..vec_size].* = vec * inv_sum_splat;
    }
    
    for (i..logits.len) |j| {
        logits[j] *= inv_sum;
    }
}

pub fn silu(x: f32) f32 {
    // SiLU activation (also called Swish)
    return x / (1.0 + @exp(-x));
}

pub fn silu_simd(x: @Vector(8, f32)) @Vector(8, f32) {
    const ones = @as(@Vector(8, f32), @splat(1.0));
    return x / (ones + @exp(-x));
}

pub fn rms_norm(x: []f32, weight: []const f32, eps: f32) void {
    // Compute RMS
    var sum_sq: f32 = 0.0;
    for (x) |val| {
        sum_sq += val * val;
    }
    
    const rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    
    // Apply normalization and weight
    for (x, weight) |*val, w| {
        val.* = val.* * rms * w;
    }
}

test "math - exp approximation" {
    const x: f32 = 1.0;
    const result = exp(x);
    const expected = std.math.exp(x);
    const diff = @abs(result - expected);
    
    // Now using built-in exp, this should be exact
    try std.testing.expect(diff < 1e-6);
}

test "math - softmax stability" {
    var logits = [_]f32{ 1000.0, 1001.0, 1002.0, 1003.0 };
    softmax(&logits);
    
    // Check that sum is 1
    var sum: f32 = 0.0;
    for (logits) |val| {
        sum += val;
    }
    try std.testing.expect(@abs(sum - 1.0) < 1e-6);
}

test "math - silu" {
    const x: f32 = 0.0;
    const result = silu(x);
    try std.testing.expect(@abs(result - 0.0) < 1e-6);
    
    const x2: f32 = 1.0;
    const result2 = silu(x2);
    try std.testing.expect(result2 > 0.7 and result2 < 0.8);
}
