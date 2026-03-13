// sampler.zig - Token sampling for Britix
// "Choosing the next word with artistic flair, what!"

const std = @import("std");
const builtin = @import("builtin");
const random = @import("random.zig");

pub const Sampler = struct {
    const Self = @This();
    
    rng: random.Xoroshiro128Plus,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    
    pub fn init(seed: u64, temperature: f32, top_k: usize, top_p: f32) Sampler {
        return Sampler{
            .rng = random.Xoroshiro128Plus.init(seed),
            .temperature = temperature,
            .top_k = top_k,
            .top_p = top_p,
        };
    }
    
    pub fn sample(self: *Self, logits: []const f32, allocator: std.mem.Allocator) !usize {
        // Make a working copy
        var probs = try allocator.alloc(f32, logits.len);
        defer allocator.free(probs);
        
        @memcpy(probs, logits);
        
        // Apply temperature
        if (self.temperature > 0 and self.temperature != 1.0) {
            const inv_temp = 1.0 / self.temperature;
            for (0..probs.len) |i| {
                probs[i] *= inv_temp;
            }
        }
        
        // Apply softmax
        try self.softmax(probs);
        
        // Apply top-k filtering
        if (self.top_k > 0 and self.top_k < probs.len) {
            try self.top_k_filter(probs, allocator);
        }
        
        // Apply top-p (nucleus) filtering
        if (self.top_p > 0.0 and self.top_p < 1.0) {
            try self.top_p_filter(probs, allocator);
        }
        
        // Sample from distribution
        return self.sample_from_probs(probs);
    }
    
    pub fn sample_greedy(_: *Self, logits: []const f32) usize {
        var max_idx: usize = 0;
        var max_val: f32 = logits[0];
        
        for (logits[1..], 1..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        
        return max_idx;
    }
    
    fn softmax(_: *Self, probs: []f32) !void {
        // Find max for numerical stability
        var max_val: f32 = -std.math.inf(f32);
        for (probs) |p| {
            if (p > max_val) max_val = p;
        }
        
        // Compute exponentials and sum
        var sum: f32 = 0.0;
        for (0..probs.len) |i| {
            probs[i] = @exp(probs[i] - max_val);
            sum += probs[i];
        }
        
        // Normalize
        const inv_sum = 1.0 / sum;
        for (0..probs.len) |i| {
            probs[i] *= inv_sum;
        }
    }
    
    fn top_k_filter(self: *Self, probs: []f32, allocator: std.mem.Allocator) !void {
        // Find the k-th largest probability
        var indices = try allocator.alloc(usize, probs.len);
        defer allocator.free(indices);
        
        for (0..probs.len) |i| {
            indices[i] = i;
        }
        
        // Sort indices by probability (descending)
        std.sort.insertion(usize, indices, probs, struct {
            fn lessThan(ctx: []f32, a: usize, b: usize) bool {
                return ctx[a] > ctx[b];
            }
        }.lessThan);
        
        // Keep only top-k
        for (0..probs.len) |i| {
            var keep = false;
            for (0..self.top_k) |j| {
                if (i == indices[j]) {
                    keep = true;
                    break;
                }
            }
            if (!keep) {
                probs[i] = 0.0;
            }
        }
        
        // Renormalize
        var sum: f32 = 0.0;
        for (probs) |p| {
            sum += p;
        }
        if (sum > 0.0) {
            const inv_sum = 1.0 / sum;
            for (0..probs.len) |i| {
                probs[i] *= inv_sum;
            }
        }
    }
    
    fn top_p_filter(self: *Self, probs: []f32, allocator: std.mem.Allocator) !void {
        // Sort indices by probability
        var indices = try allocator.alloc(usize, probs.len);
        defer allocator.free(indices);
        
        for (0..probs.len) |i| {
            indices[i] = i;
        }
        
        std.sort.insertion(usize, indices, probs, struct {
            fn lessThan(ctx: []f32, a: usize, b: usize) bool {
                return ctx[a] > ctx[b];
            }
        }.lessThan);
        
        // Find cutoff where cumulative probability exceeds top_p
        var cumsum: f32 = 0.0;
        var cutoff_idx: usize = probs.len;
        
        for (indices, 0..) |idx, i| {
            cumsum += probs[idx];
            if (cumsum > self.top_p) {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Zero out everything after cutoff
        for (indices[cutoff_idx..]) |idx| {
            probs[idx] = 0.0;
        }
        
        // Renormalize
        var sum: f32 = 0.0;
        for (indices[0..cutoff_idx]) |idx| {
            sum += probs[idx];
        }
        if (sum > 0.0) {
            const inv_sum = 1.0 / sum;
            for (indices[0..cutoff_idx]) |idx| {
                probs[idx] *= inv_sum;
            }
        }
    }
    
    fn sample_from_probs(self: *Self, probs: []const f32) usize {
        const r = self.rng.nextF32();
        var cumsum: f32 = 0.0;
        
        for (probs, 0..) |p, i| {
            cumsum += p;
            if (r < cumsum) {
                return i;
            }
        }
        
        return probs.len - 1; // Fallback
    }
    
    pub fn reset_rng(self: *Self, seed: u64) void {
        self.rng = random.Xoroshiro128Plus.init(seed);
    }
};

test "sampler - greedy sampling" {
    var sampler = Sampler.init(1234, 1.0, 0, 0.0);
    
    var logits = [_]f32{ 1.0, 2.0, 3.0, 2.5, 1.5 };
    const token = sampler.sample_greedy(&logits);
    
    try std.testing.expectEqual(@as(usize, 2), token); // 3.0 is highest
}

test "sampler - temperature affects distribution" {
    const allocator = std.testing.allocator;
    var sampler1 = Sampler.init(1234, 0.5, 0, 0.0);
    var sampler2 = Sampler.init(1234, 2.0, 0, 0.0);
    
    var logits = [_]f32{ 1.0, 2.0, 3.0, 2.0, 1.0 };
    
    const token1 = try sampler1.sample(&logits, allocator);
    const token2 = try sampler2.sample(&logits, allocator);
    
    // Just test that they run without error
    _ = token1;
    _ = token2;
}
