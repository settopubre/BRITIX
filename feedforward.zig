// feedforward.zig - Feed-forward network for Britix (SwiGLU)
// "Processing tokens with British efficiency, what!"

const std = @import("std");
const builtin = @import("builtin");

const parameters = @import("parameters.zig");
const math = @import("math.zig");
const simd = @import("simd.zig");

pub const FeedForwardConfig = struct {
    dim: usize,
    hidden_dim: usize,
    multiple_of: usize = 256,  // For efficient computation
    activation: enum { gelu, silu, swiglu } = .swiglu,
    use_quantized: bool = false,
};

pub const FeedForward = struct {
    const Self = @This();
    
    config: FeedForwardConfig,
    layer_idx: usize,
    params: *parameters.Parameters,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, params: *parameters.Parameters, layer_idx: usize, dim: usize, hidden_dim: usize) !FeedForward {
        return FeedForward{
            .config = FeedForwardConfig{
                .dim = dim,
                .hidden_dim = hidden_dim,
            },
            .layer_idx = layer_idx,
            .params = params,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *FeedForward) void {
        _ = self;
    }
    
    pub fn forward(self: *Self, x: []const f32) ![]f32 {
        const config = self.config;
        const layer = &self.params.layers[self.layer_idx];
        const B = 1; // Batch size
        const T = x.len / config.dim;
        
        // Allocate output
        const output = try self.allocator.alloc(f32, B * T * config.dim);
        errdefer self.allocator.free(output);
        
        // SwiGLU: FFN = (silu(x @ w1)) * (x @ w3) @ w2
        // w1: gate, w3: up, w2: down
        
        // Allocate gate and up
        const gate = try self.allocator.alloc(f32, B * T * config.hidden_dim);
        defer self.allocator.free(gate);
        const up = try self.allocator.alloc(f32, B * T * config.hidden_dim);
        defer self.allocator.free(up);
        
        // Gate projection: x @ w1
        try self.linear(x, layer, .w1, gate, config.dim, config.hidden_dim);
        
        // Up projection: x @ w3
        try self.linear(x, layer, .w3, up, config.dim, config.hidden_dim);
        
        // Apply SiLU activation to gate
        for (0..B * T * config.hidden_dim) |i| {
            gate[i] = math.silu(gate[i]);
        }
        
        // Element-wise multiply: gate * up
        for (0..B * T * config.hidden_dim) |i| {
            gate[i] = gate[i] * up[i];
        }
        
        // Down projection: (gate * up) @ w2
        try self.linear(gate, layer, .w2, output, config.hidden_dim, config.dim);
        
        return output;
    }
    
    fn linear(self: *Self, input: []const f32, layer: *parameters.MappedLayer, weight_type: enum { w1, w2, w3 }, output: []f32, in_dim: usize, out_dim: usize) !void {
        const batch_seq = input.len / in_dim;
        
        for (0..batch_seq) |bs| {
            const in_offset = bs * in_dim;
            const out_offset = bs * out_dim;
            
            for (0..out_dim) |o| {
                var sum: f32 = 0.0;
                for (0..in_dim) |i| {
                    const weight_val = switch (weight_type) {
                        .w1 => try layer.w1.get2D(
                            self.params.mapped_data,
                            @as(u32, @intCast(i)),
                            @as(u32, @intCast(o))
                        ),
                        .w2 => try layer.w2.get2D(
                            self.params.mapped_data,
                            @as(u32, @intCast(i)),
                            @as(u32, @intCast(o))
                        ),
                        .w3 => try layer.w3.get2D(
                            self.params.mapped_data,
                            @as(u32, @intCast(i)),
                            @as(u32, @intCast(o))
                        ),
                    };
                    sum += input[in_offset + i] * weight_val;
                }
                output[out_offset + o] = sum;
            }
        }
    }
    
    pub fn swiglu(self: *Self, x: f32) f32 {
        _ = self;
        // SiLU (Sigmoid Linear Unit) = x * sigmoid(x)
        return math.silu(x);
    }
    
    pub fn compute_hidden_dim(comptime dim: usize, comptime multiple_of: usize) usize {
        // For Mistral 7B / Llama architecture:
        // hidden_dim = (4 * dim * 2/3) rounded to nearest multiple_of
        var hidden = (4 * dim * 2) / 3;
        
        // Round up to nearest multiple_of
        hidden = ((hidden + multiple_of - 1) / multiple_of) * multiple_of;
        
        return hidden;
    }
};

test "feedforward - hidden dim calculation" {
    const dim = 4096;
    const hidden = FeedForward.compute_hidden_dim(dim, 256);
    
    // Should be 11008 for Mistral 7B (not 14336)
    try std.testing.expectEqual(@as(usize, 11008), hidden);
}

test "feedforward - swiglu activation" {
    var ff = FeedForward{
        .config = undefined,
        .layer_idx = 0,
        .params = undefined,
        .allocator = std.testing.allocator,
    };
    
    // SiLU(0) should be 0
    try std.testing.expectApproxEqAbs(0.0, ff.swiglu(0.0), 0.001);
    
    // SiLU(1) ~ 0.731
    try std.testing.expectApproxEqAbs(0.731, ff.swiglu(1.0), 0.01);
}

test "feedforward - linear with mock params" {
    const allocator = std.testing.allocator;
    
    // Create minimal config
    const config = parameters.BritixConfig{
        .dim = 2048,
        .n_layers = 36,
        .n_heads = 16,
        .n_kv_heads = 2,
        .vocab_size = 151936,
        .hidden_dim = 5632,
        .max_seq_len = 32768,
        .norm_eps = 1e-6,
        .rope_theta = 1000000.0,
    };
    
    // Create mock data
    const mock_data = try allocator.alloc(u8, 100 * 1024 * 1024);
    defer allocator.free(mock_data);
    @memset(mock_data, 0);
    
    // Create parameters
    var params = parameters.Parameters{
        .config = config,
        .mapped_data = mock_data,
        .token_embedding = undefined,
        .layers = undefined,
        .norm_weight = undefined,
    };
    
    // FIXED: Removed the ", "test_norm" string parameter
    params.token_embedding = parameters.MappedTensor.init(0, &[_]u32{ config.vocab_size, config.dim });
    params.norm_weight = parameters.MappedTensor.init(0, &[_]u32{config.dim});
    
    params.layers = try allocator.alloc(parameters.MappedLayer, 1);
    defer allocator.free(params.layers);
    
    var offset: u64 = 0;
    params.layers[0] = parameters.MappedLayer.init(&offset, config, 0);
    
    var ff = try FeedForward.init(allocator, &params, 0, 512, 1024);
    defer ff.deinit();
    
    const x = try allocator.alloc(f32, 512);
    defer allocator.free(x);
    @memset(x, 1.0);
    
    const output = try ff.forward(x);
    defer allocator.free(output);
    
    try std.testing.expect(output.len == 512);
}
