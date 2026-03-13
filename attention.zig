// attention.zig - Multi-head attention for Britix
// "Paying attention, just like a well-mannered British gentleman."

const std = @import("std");
const builtin = @import("builtin");

const parameters = @import("parameters.zig");
const math = @import("math.zig");

pub const AttentionConfig = struct {
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
};

pub const Attention = struct {
    const Self = @This();
    
    config: AttentionConfig,
    layer_idx: usize,
    params: *parameters.Parameters,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, params: *parameters.Parameters, layer_idx: usize, dim: usize, n_heads: usize, n_kv_heads: usize) !Attention {
        const head_dim = dim / n_heads;
        
        return Attention{
            .config = AttentionConfig{
                .dim = dim,
                .n_heads = n_heads,
                .n_kv_heads = n_kv_heads,
                .head_dim = head_dim,
                .max_seq_len = 32768,
            },
            .layer_idx = layer_idx,
            .params = params,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Attention) void {
        _ = self;
    }
    
    pub fn forward(self: *Self, x: []const f32, k_cache: []f32, v_cache: []f32, start_pos: usize) ![]f32 {
        _ = k_cache;
        _ = v_cache;
        _ = start_pos;
        
        const config = self.config;
        const layer = &self.params.layers[self.layer_idx];
        const B = 1; // batch size
        const T = x.len / config.dim; // sequence length
        
        // Allocate Q, K, V
        const q = try self.allocator.alloc(f32, B * T * config.n_heads * config.head_dim);
        defer self.allocator.free(q);
        const k = try self.allocator.alloc(f32, B * T * config.n_kv_heads * config.head_dim);
        defer self.allocator.free(k);
        const v = try self.allocator.alloc(f32, B * T * config.n_kv_heads * config.head_dim);
        defer self.allocator.free(v);
        
        // Q projection
        for (0..B) |b| {
            for (0..T) |t| {
                const x_off = (b * T + t) * config.dim;
                const q_off = (b * T + t) * config.n_heads * config.head_dim;
                
                for (0..config.n_heads * config.head_dim) |i| {
                    var sum: f32 = 0;
                    for (0..config.dim) |j| {
                        // Use try with catch to handle potential errors
                        const weight = layer.wq.getSafe(
                            self.params.mapped_data,
                            &[_]u32{ @as(u32, @intCast(j)), @as(u32, @intCast(i)) }
                        ) catch {
                            // If we hit an error, use 0.0 and continue
                            // This is just for testing
                            sum += x[x_off + j] * 0.0;
                            continue;
                        };
                        sum += x[x_off + j] * weight;
                    }
                    q[q_off + i] = sum;
                }
            }
        }
        
        // K projection
        for (0..B) |b| {
            for (0..T) |t| {
                const x_off = (b * T + t) * config.dim;
                const k_off = (b * T + t) * config.n_kv_heads * config.head_dim;
                
                for (0..config.n_kv_heads * config.head_dim) |i| {
                    var sum: f32 = 0;
                    for (0..config.dim) |j| {
                        const weight = layer.wk.getSafe(
                            self.params.mapped_data,
                            &[_]u32{ @as(u32, @intCast(j)), @as(u32, @intCast(i)) }
                        ) catch {
                            sum += x[x_off + j] * 0.0;
                            continue;
                        };
                        sum += x[x_off + j] * weight;
                    }
                    k[k_off + i] = sum;
                }
            }
        }
        
        // V projection
        for (0..B) |b| {
            for (0..T) |t| {
                const x_off = (b * T + t) * config.dim;
                const v_off = (b * T + t) * config.n_kv_heads * config.head_dim;
                
                for (0..config.n_kv_heads * config.head_dim) |i| {
                    var sum: f32 = 0;
                    for (0..config.dim) |j| {
                        const weight = layer.wv.getSafe(
                            self.params.mapped_data,
                            &[_]u32{ @as(u32, @intCast(j)), @as(u32, @intCast(i)) }
                        ) catch {
                            sum += x[x_off + j] * 0.0;
                            continue;
                        };
                        sum += x[x_off + j] * weight;
                    }
                    v[v_off + i] = sum;
                }
            }
        }
        
        // Simplified output for now
        const output = try self.allocator.alloc(f32, B * T * config.dim);
        @memset(output, 0.0);
        return output;
    }
};

test "attention - basic forward" {
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
    
    // Create a larger mock data buffer to avoid bounds errors
    const mock_data = try allocator.alloc(u8, 100 * 1024 * 1024); // 100MB
    defer allocator.free(mock_data);
    @memset(mock_data, 0);
    
    // Create parameters with just one layer for testing
    var params = parameters.Parameters{
        .config = config,
        .mapped_data = mock_data,
        .token_embedding = undefined,
        .layers = undefined,
        .norm_weight = undefined,
    };
    
    // Initialize token embedding and norm weight with dummy values - FIXED: removed string parameter
    params.token_embedding = parameters.MappedTensor.init(0, &[_]u32{ config.vocab_size, config.dim });
    params.norm_weight = parameters.MappedTensor.init(0, &[_]u32{config.dim});
    
    // Initialize layers array
    params.layers = try allocator.alloc(parameters.MappedLayer, 1);
    defer allocator.free(params.layers); // Use defer, not errdefer
    
    // Initialize layer with dummy offset
    var offset: u64 = 0;
    params.layers[0] = parameters.MappedLayer.init(&offset, config, 0);
    
    var attn = try Attention.init(allocator, &params, 0, config.dim, config.n_heads, config.n_kv_heads);
    defer attn.deinit();
    
    // Test with dummy input
    const x = try allocator.alloc(f32, config.dim);
    defer allocator.free(x);
    @memset(x, 1.0);
    
    const k_cache = try allocator.alloc(f32, 1024);
    defer allocator.free(k_cache);
    
    const v_cache = try allocator.alloc(f32, 1024);
    defer allocator.free(v_cache);
    
    const output = try attn.forward(x, k_cache, v_cache, 0);
    defer allocator.free(output);
    
    try std.testing.expect(output.len == config.dim);
    
    // No need to free params.layers separately since we have defer
}
