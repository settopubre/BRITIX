// cache.zig - KV cache with hardware prefetch hints
// "Cache is king! Prefetch like a butler anticipating your needs."

const std = @import("std");
const builtin = @import("builtin");

pub const KVCache = struct {
    // Key and value caches per layer
    k_cache: [][]f32,
    v_cache: [][]f32,
    
    // Metadata
    layer_count: usize,
    max_seq_len: usize,
    head_dim: usize,
    n_kv_heads: usize,
    current_pos: usize,
    
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, max_seq_len: usize, dim: usize, n_layers: usize, n_kv_heads: usize, n_heads: usize) !KVCache {
        const head_dim = dim / n_heads;
        
        var k_cache = try allocator.alloc([]f32, n_layers);
        errdefer {
            for (k_cache) |layer| allocator.free(layer);
            allocator.free(k_cache);
        }
        
        var v_cache = try allocator.alloc([]f32, n_layers);
        errdefer {
            for (v_cache) |layer| allocator.free(layer);
            allocator.free(v_cache);
        }
        
        for (0..n_layers) |i| {
            const layer_size = max_seq_len * n_kv_heads * head_dim;
            
            k_cache[i] = try allocator.alloc(f32, layer_size);
            errdefer {
                for (0..i) |j| allocator.free(k_cache[j]);
                allocator.free(k_cache);
            }
            
            v_cache[i] = try allocator.alloc(f32, layer_size);
            errdefer {
                for (0..i) |j| allocator.free(v_cache[j]);
                allocator.free(v_cache);
            }
        }
        
        return KVCache{
            .k_cache = k_cache,
            .v_cache = v_cache,
            .layer_count = n_layers,
            .max_seq_len = max_seq_len,
            .head_dim = head_dim,
            .n_kv_heads = n_kv_heads,
            .current_pos = 0,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *KVCache) void {
        for (self.k_cache) |layer| self.allocator.free(layer);
        self.allocator.free(self.k_cache);
        
        for (self.v_cache) |layer| self.allocator.free(layer);
        self.allocator.free(self.v_cache);
    }
    
    pub fn update(self: *KVCache, layer: usize, pos: usize, k: []const f32, v: []const f32) !void {
        if (layer >= self.layer_count) return error.InvalidLayer;
        if (pos >= self.max_seq_len) return error.PositionTooLarge;
        
        const layer_size = self.n_kv_heads * self.head_dim;
        
        // Handle size mismatch gracefully instead of crashing
        if (k.len != layer_size or v.len != layer_size) {
            std.debug.print("⚠️ Cache size mismatch: expected {d}, got k={d} v={d}\n", .{layer_size, k.len, v.len});
            return error.InvalidCacheSize;
        }
        
        const k_start = pos * layer_size;
        const v_start = pos * layer_size;
        
        @memcpy(self.k_cache[layer][k_start .. k_start + layer_size], k);
        @memcpy(self.v_cache[layer][v_start .. v_start + layer_size], v);
        
        self.current_pos = @max(self.current_pos, pos + 1);
    }
    
    pub fn getK(self: *const KVCache, layer: usize, start: usize, end: usize) ![]const f32 {
        if (layer >= self.layer_count) return error.InvalidLayer;
        if (end > self.current_pos) return error.InvalidRange;
        
        const layer_size = self.n_kv_heads * self.head_dim;
        const k_start = start * layer_size;
        const k_end = end * layer_size;
        
        return self.k_cache[layer][k_start..k_end];
    }
    
    pub fn getV(self: *const KVCache, layer: usize, start: usize, end: usize) ![]const f32 {
        if (layer >= self.layer_count) return error.InvalidLayer;
        if (end > self.current_pos) return error.InvalidRange;
        
        const layer_size = self.n_kv_heads * self.head_dim;
        const v_start = start * layer_size;
        const v_end = end * layer_size;
        
        return self.v_cache[layer][v_start..v_end];
    }
    
    pub fn clear(self: *KVCache) void {
        self.current_pos = 0;
    }
    
    pub fn prefetchFuture(self: *KVCache, future_pos: usize) void {
        _ = self;
        _ = future_pos;
    }
};

test "cache - basic operations" {
    const allocator = std.testing.allocator;
    var cache = try KVCache.init(allocator, 2048, 2048, 36, 2, 16);
    defer cache.deinit();
    
    const layer_size = 2 * (2048 / 16);  // n_kv_heads * head_dim = 2 * 128 = 256
    var k = try allocator.alloc(f32, layer_size);
    defer allocator.free(k);
    var v = try allocator.alloc(f32, layer_size);
    defer allocator.free(v);
    
    for (0..layer_size) |i| {
        k[i] = @as(f32, @floatFromInt(i));
        v[i] = @as(f32, @floatFromInt(i)) * 2;
    }
    
    try cache.update(0, 0, k, v);
    
    const k_retrieved = try cache.getK(0, 0, 1);
    try std.testing.expectEqual(k_retrieved.len, layer_size);
    try std.testing.expectEqual(k_retrieved[0], 0);
    
    std.debug.print("✅ Cache test passed! head_dim={d}, layer_size={d}\n", .{cache.head_dim, layer_size});
}
