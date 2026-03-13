// embed.zig - Embedding extraction for Britix
// "Getting to the essence of words. Deep stuff, what!"

const std = @import("std");
const builtin = @import("builtin");

pub const EmbeddingType = enum {
    token_embeddings,
    last_hidden_state,
    pooled_output,
    cls_token,
};

pub const EmbeddingConfig = struct {
    embedding_type: EmbeddingType = .last_hidden_state,
    normalize: bool = true,
    layer_idx: ?usize = null,
    pooling_strategy: enum { mean, max, cls } = .mean,
};

pub const EmbeddingEngine = struct {
    const Self = @This();
    
    config: EmbeddingConfig,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .config = EmbeddingConfig{},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        _ = self;
    }
    
    pub fn similarity(self: *Self, a: []const f32, b: []const f32) f32 {
        _ = self;
        // Cosine similarity
        var dot: f32 = 0.0;
        var norm_a: f32 = 0.0;
        var norm_b: f32 = 0.0;
        
        for (0..a.len) |i| {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        if (norm_a == 0.0 or norm_b == 0.0) return 0.0;
        return dot / (@sqrt(norm_a) * @sqrt(norm_b));
    }
};

test "embed - similarity" {
    const allocator = std.testing.allocator;
    var engine = EmbeddingEngine.init(allocator);
    defer engine.deinit();
    
    const a = try allocator.alloc(f32, 3);
    defer allocator.free(a);
    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 3.0;
    
    const b = try allocator.alloc(f32, 3);
    defer allocator.free(b);
    b[0] = 1.0;
    b[1] = 2.0;
    b[2] = 3.0;
    
    const sim = engine.similarity(a, b);
    try std.testing.expectApproxEqAbs(1.0, sim, 0.001);
}

test "embed - similarity orthogonal" {
    const allocator = std.testing.allocator;
    var engine = EmbeddingEngine.init(allocator);
    defer engine.deinit();
    
    const a = try allocator.alloc(f32, 2);
    defer allocator.free(a);
    a[0] = 1.0;
    a[1] = 0.0;
    
    const b = try allocator.alloc(f32, 2);
    defer allocator.free(b);
    b[0] = 0.0;
    b[1] = 1.0;
    
    const sim = engine.similarity(a, b);
    try std.testing.expectApproxEqAbs(0.0, sim, 0.001);
}
