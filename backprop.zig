// backprop.zig - Automatic differentiation for Britix
// "Calculus with a British accent. Mind the gradients!"

const std = @import("std");
const builtin = @import("builtin");

const parameters = @import("parameters.zig");

pub const GradientBuffer = struct {
    const Self = @This();
    
    // Gradients for all parameters (simplified - just one buffer for testing)
    data: []f32,
    size: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, size: usize) !Self {
        return Self{
            .data = try allocator.alloc(f32, size),
            .size = size,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
    }
    
    pub fn clear(self: *Self) void {
        @memset(self.data, 0.0);
    }
    
    pub fn add(self: *Self, other: *const GradientBuffer) !void {
        for (0..self.size) |i| {
            self.data[i] += other.data[i];
        }
    }
    
    pub fn clip(self: *Self, max_norm: f32) !void {
        // Compute global norm
        var global_norm: f32 = 0.0;
        for (self.data) |g| {
            global_norm += g * g;
        }
        global_norm = @sqrt(global_norm);
        
        if (global_norm > max_norm) {
            const scale = max_norm / global_norm;
            for (self.data) |*g| {
                g.* *= scale;
            }
        }
    }
};

pub const Context = struct {
    const Self = @This();
    
    params: *parameters.Parameters,
    grads: GradientBuffer,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, params: *parameters.Parameters) !Self {
        return Self{
            .params = params,
            .grads = try GradientBuffer.init(allocator, 100), // Small test size
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.grads.deinit();
    }
    
    pub fn cross_entropy_loss(self: *Self, logits: []const f32, targets: []const u32) !f32 {
        _ = self;
        _ = logits;
        _ = targets;
        return 1.0; // Dummy loss for testing
    }
};

test "backprop - gradient buffer" {
    const allocator = std.testing.allocator;
    
    var grads = try GradientBuffer.init(allocator, 100);
    defer grads.deinit();
    
    // Set some values
    for (0..100) |i| {
        grads.data[i] = @as(f32, @floatFromInt(i)) / 100.0;
    }
    
    grads.clear();
    
    // Check that clear worked
    for (grads.data) |val| {
        try std.testing.expectEqual(@as(f32, 0.0), val);
    }
}

test "backprop - gradient clip" {
    const allocator = std.testing.allocator;
    
    var grads = try GradientBuffer.init(allocator, 10);
    defer grads.deinit();
    
    // Set large values
    for (0..10) |i| {
        grads.data[i] = 10.0;
    }
    
    const before_norm = blk: {
        var sum: f32 = 0.0;
        for (grads.data) |g| sum += g * g;
        break :blk @sqrt(sum);
    };
    
    try grads.clip(5.0);
    
    const after_norm = blk: {
        var sum: f32 = 0.0;
        for (grads.data) |g| sum += g * g;
        break :blk @sqrt(sum);
    };
    
    try std.testing.expect(after_norm <= 5.0 + 0.001);
    try std.testing.expect(after_norm < before_norm);
}

test "backprop - context init" {
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
    
    // Create dummy mapped data
    const mock_data = try allocator.alloc(u8, 1024);
    defer allocator.free(mock_data);
    @memset(mock_data, 0);
    
    // Create parameters without allocator field
    var params = parameters.Parameters{
        .config = config,
        .mapped_data = mock_data,
        .token_embedding = undefined,
        .layers = undefined,
        .norm_weight = undefined,
    };
    
    // Initialize token embedding and norm weight with dummy values - FIXED: removed "test_norm"
    params.token_embedding = parameters.MappedTensor.init(0, &[_]u32{ config.vocab_size, config.dim });
    params.norm_weight = parameters.MappedTensor.init(0, &[_]u32{config.dim});
    
    // Initialize layers array with one layer
    params.layers = try allocator.alloc(parameters.MappedLayer, 1);
    defer allocator.free(params.layers);
    
    var offset: u64 = 0;
    params.layers[0] = parameters.MappedLayer.init(&offset, config, 0);
    
    var ctx = try Context.init(allocator, &params);
    defer ctx.deinit();
    
    try std.testing.expect(ctx.grads.size == 100);
}
