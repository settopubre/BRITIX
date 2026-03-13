// optimizer.zig - AdamW optimizer for Britix
// "Shaping the weights with mathematical precision, what!"

const std = @import("std");
const builtin = @import("builtin");

const parameters = @import("parameters.zig");
const backprop = @import("backprop.zig");

pub const OptimizerConfig = struct {
    learning_rate: f32 = 3e-4,
    beta1: f32 = 0.9,
    beta2: f32 = 0.95,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.1,
    use_fused: bool = true,
};

pub const Optimizer = struct {
    const Self = @This();
    
    config: OptimizerConfig,
    params: *parameters.Parameters,
    
    // Adam state (simplified - just single buffers for testing)
    m: backprop.GradientBuffer,
    v: backprop.GradientBuffer,
    
    step: usize = 0,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, params: *parameters.Parameters, config: OptimizerConfig) !Self {
        return Self{
            .config = config,
            .params = params,
            .m = try backprop.GradientBuffer.init(allocator, 100), // Small test size
            .v = try backprop.GradientBuffer.init(allocator, 100),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.m.deinit();
        self.v.deinit();
    }
    
    pub fn step(self: *Self, grads: *const backprop.GradientBuffer, lr: f32) !void {
        self.step += 1;
        
        const bias_correction1 = 1.0 - std.math.pow(f32, self.config.beta1, @as(f32, @floatFromInt(self.step)));
        const bias_correction2 = 1.0 - std.math.pow(f32, self.config.beta2, @as(f32, @floatFromInt(self.step)));
        
        // Simple parameter update for testing
        try self.update_params(grads, lr, bias_correction1, bias_correction2);
    }
    
    fn update_params(self: *Self, grads: *const backprop.GradientBuffer, lr: f32, bc1: f32, bc2: f32) !void {
        const beta1 = self.config.beta1;
        const beta2 = self.config.beta2;
        const eps = self.config.eps;
        const wd = self.config.weight_decay;
        
        for (0..self.m.size) |i| {
            // Update biased first moment estimate
            self.m.data[i] = beta1 * self.m.data[i] + (1.0 - beta1) * grads.data[i];
            
            // Update biased second raw moment estimate
            self.v.data[i] = beta2 * self.v.data[i] + (1.0 - beta2) * grads.data[i] * grads.data[i];
            
            // Compute bias-corrected estimates
            const m_hat = self.m.data[i] / bc1;
            const v_hat = self.v.data[i] / bc2;
            
            // Dummy parameter update (no actual params for testing)
            _ = m_hat;
            _ = v_hat;
            _ = wd;
            _ = eps;
            _ = lr;
        }
    }
    
    pub fn zero_grad(self: *Self) void {
        self.m.clear();
        self.v.clear();
    }
};

test "optimizer - initialization" {
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
    
    // Create dummy params (without allocator field)
    var params = parameters.Parameters{
        .config = config,
        .mapped_data = mock_data,
        .token_embedding = undefined,
        .layers = undefined,
        .norm_weight = undefined,
    };
    
    // Initialize required fields - FIXED: removed ", "test_norm" strings
    params.token_embedding = parameters.MappedTensor.init(0, &[_]u32{ config.vocab_size, config.dim });
    params.norm_weight = parameters.MappedTensor.init(0, &[_]u32{config.dim});
    
    // Initialize layers array
    params.layers = try allocator.alloc(parameters.MappedLayer, 1);
    defer allocator.free(params.layers);
    
    var offset: u64 = 0;
    params.layers[0] = parameters.MappedLayer.init(&offset, config, 0);
    
    var opt = try Optimizer.init(allocator, &params, .{});
    defer opt.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), opt.step);
    try std.testing.expectEqual(@as(usize, 100), opt.m.size);
    try std.testing.expectEqual(@as(usize, 100), opt.v.size);
}

test "optimizer - zero grad" {
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
    
    // Create dummy params (without allocator field)
    var params = parameters.Parameters{
        .config = config,
        .mapped_data = mock_data,
        .token_embedding = undefined,
        .layers = undefined,
        .norm_weight = undefined,
    };
    
    // Initialize required fields - FIXED: removed ", "test_norm" strings
    params.token_embedding = parameters.MappedTensor.init(0, &[_]u32{ config.vocab_size, config.dim });
    params.norm_weight = parameters.MappedTensor.init(0, &[_]u32{config.dim});
    
    // Initialize layers array
    params.layers = try allocator.alloc(parameters.MappedLayer, 1);
    defer allocator.free(params.layers);
    
    var offset: u64 = 0;
    params.layers[0] = parameters.MappedLayer.init(&offset, config, 0);
    
    var opt = try Optimizer.init(allocator, &params, .{});
    defer opt.deinit();
    
    // Set some values
    for (0..100) |i| {
        opt.m.data[i] = @as(f32, @floatFromInt(i));
        opt.v.data[i] = @as(f32, @floatFromInt(i));
    }
    
    opt.zero_grad();
    
    // Check that zero_grad cleared
    for (0..100) |i| {
        try std.testing.expectEqual(@as(f32, 0.0), opt.m.data[i]);
        try std.testing.expectEqual(@as(f32, 0.0), opt.v.data[i]);
    }
}
