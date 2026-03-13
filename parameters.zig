const std = @import("std");

pub const BritixConfig = struct {
    dim: u32 = 2048,           // Qwen2.5 3B uses 2048 hidden size
    n_layers: u32 = 36,        // 36 layers
    n_heads: u32 = 16,         // 16 attention heads
    n_kv_heads: u32 = 2,       // GQA: 2 KV heads
    vocab_size: u32 = 151936,  // Qwen2.5 vocab size
    hidden_dim: u32 = 5632,    // FFN intermediate size
    max_seq_len: u32 = 32768,  // Context length
    norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1000000.0,
    
    pub fn totalParams(self: *const BritixConfig) u64 {
        var total: u64 = @as(u64, self.vocab_size) * @as(u64, self.dim);
        const attn_params = 4 * @as(u64, self.dim) * @as(u64, self.dim);
        const ff_params = 3 * @as(u64, self.dim) * @as(u64, self.hidden_dim);
        const layer_params = attn_params + ff_params + 2 * @as(u64, self.dim);
        total += layer_params * @as(u64, self.n_layers);
        total += @as(u64, self.dim);
        return total;
    }
    
    pub fn validate(self: *const BritixConfig) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.n_layers == 0) return error.InvalidLayers;
        if (self.n_heads == 0) return error.InvalidHeads;
        if (self.vocab_size == 0) return error.InvalidVocab;
        if (self.hidden_dim == 0) return error.InvalidHiddenDim;
        
        if (self.dim > 65536) return error.DimensionTooLarge;
        if (self.n_layers > 256) return error.TooManyLayers;
    }
};

// Memory-mapped tensor - NO NAME FIELD to prevent corruption
pub const MappedTensor = struct {
    offset: u64,
    shape: ?[]const u32,
    strides: ?[]const u32,
    elem_size: u32,
    total_elements: u64,
    
    pub fn init(offset: u64, shape: []const u32) MappedTensor {
        var total: u64 = 1;
        for (shape) |dim| total *= dim;
        
        var result = MappedTensor{
            .offset = offset,
            .shape = null,
            .strides = null,
            .elem_size = 2,
            .total_elements = total,
        };
        
        const strides = std.heap.page_allocator.alloc(u32, shape.len) catch return result;
        
        var stride: u32 = 1;
        var i: usize = shape.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= shape[i];
        }
        
        const shape_copy = std.heap.page_allocator.dupe(u32, shape) catch {
            std.heap.page_allocator.free(strides);
            return result;
        };
        
        result.shape = shape_copy;
        result.strides = strides;
        return result;
    }
    
    pub fn deinit(self: *MappedTensor) void {
        if (self.shape) |shape| {
            std.heap.page_allocator.free(shape);
            self.shape = null;
        }
        if (self.strides) |strides| {
            std.heap.page_allocator.free(strides);
            self.strides = null;
        }
    }
    
    // COMPLETELY SILENT getter - no prints, no names
    pub fn getSafe(self: *MappedTensor, mapped_data: []const u8, indices: []const u32) !f32 {
        const shape = self.shape orelse return error.TensorNotInitialized;
        const strides = self.strides orelse return error.TensorNotInitialized;
        
        if (indices.len != shape.len) return error.InvalidIndices;
        
        var flat_idx: u64 = 0;
        for (indices, 0..) |idx, i| {
            if (idx >= shape[i]) return error.IndexOutOfBounds;
            flat_idx += @as(u64, idx) * strides[i];
        }
        
        if (flat_idx >= self.total_elements) return error.IndexOutOfBounds;
        
        const byte_offset = self.offset + flat_idx / 2;
        if (byte_offset >= mapped_data.len) return error.OutOfMemoryBounds;
        
        const byte = mapped_data[byte_offset];
        const nibble = if (flat_idx % 2 == 0) (byte & 0x0F) else (byte >> 4) & 0x0F;
        
        return (@as(f32, @floatFromInt(nibble)) / 7.5) - 1.0;
    }
    
    pub fn get1D(self: *MappedTensor, mapped_data: []const u8, idx: u32) !f32 {
        return self.getSafe(mapped_data, &[_]u32{idx});
    }
    
    pub fn get2D(self: *MappedTensor, mapped_data: []const u8, i: u32, j: u32) !f32 {
        return self.getSafe(mapped_data, &[_]u32{i, j});
    }
};

// Layer with mapped tensors - NO NAMES
pub const MappedLayer = struct {
    wq: MappedTensor,
    wk: MappedTensor,
    wv: MappedTensor,
    wo: MappedTensor,
    attention_norm: MappedTensor,
    w1: MappedTensor,
    w2: MappedTensor,
    w3: MappedTensor,
    ffn_norm: MappedTensor,
    layer_idx: usize,
    
    pub fn init(offset: *u64, config: BritixConfig, layer_idx: usize) MappedLayer {
        const head_dim = config.dim / config.n_heads;  // 2048/16 = 128
        const kv_dim = config.n_kv_heads * head_dim;   // 2*128 = 256
        
        const wq_size = config.dim * config.dim;                    // 2048*2048
        const wk_size = kv_dim * config.dim;                        // 256*2048
        const wv_size = kv_dim * config.dim;                        // 256*2048
        const wo_size = config.dim * config.dim;                    // 2048*2048
        const norm_size = config.dim;                               // 2048
        const w1_size = config.hidden_dim * config.dim;            // 5632*2048
        const w2_size = config.dim * config.hidden_dim;            // 2048*5632
        const w3_size = config.hidden_dim * config.dim;            // 5632*2048
        
        const wq = MappedTensor.init(offset.*, &[_]u32{ config.dim, config.dim });
        offset.* += wq_size / 2;
        
        const wk = MappedTensor.init(offset.*, &[_]u32{ kv_dim, config.dim });
        offset.* += wk_size / 2;
        
        const wv = MappedTensor.init(offset.*, &[_]u32{ kv_dim, config.dim });
        offset.* += wv_size / 2;
        
        const wo = MappedTensor.init(offset.*, &[_]u32{ config.dim, config.dim });
        offset.* += wo_size / 2;
        
        const attention_norm = MappedTensor.init(offset.*, &[_]u32{norm_size});
        offset.* += norm_size / 2;
        
        const w1 = MappedTensor.init(offset.*, &[_]u32{ config.hidden_dim, config.dim });
        offset.* += w1_size / 2;
        
        const w2 = MappedTensor.init(offset.*, &[_]u32{ config.dim, config.hidden_dim });
        offset.* += w2_size / 2;
        
        const w3 = MappedTensor.init(offset.*, &[_]u32{ config.hidden_dim, config.dim });
        offset.* += w3_size / 2;
        
        const ffn_norm = MappedTensor.init(offset.*, &[_]u32{norm_size});
        offset.* += norm_size / 2;
        
        return MappedLayer{
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .attention_norm = attention_norm,
            .w1 = w1,
            .w2 = w2,
            .w3 = w3,
            .ffn_norm = ffn_norm,
            .layer_idx = layer_idx,
        };
    }
    
    pub fn deinit(self: *MappedLayer) void {
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.wo.deinit();
        self.attention_norm.deinit();
        self.w1.deinit();
        self.w2.deinit();
        self.w3.deinit();
        self.ffn_norm.deinit();
    }
};

// Parameters using mapped tensors
pub const Parameters = struct {
    config: BritixConfig,
    mapped_data: []const u8,
    token_embedding: MappedTensor,
    layers: []MappedLayer,
    norm_weight: MappedTensor,
    
    pub fn init(mapped_data: []const u8, config: BritixConfig) !Parameters {
        try config.validate();
        
        var offset: u64 = 0;
        
        const token_embedding = MappedTensor.init(offset, &[_]u32{ config.vocab_size, config.dim });
        offset += @as(u64, config.vocab_size) * config.dim / 2;
        
        var layers = try std.heap.page_allocator.alloc(MappedLayer, config.n_layers);
        errdefer std.heap.page_allocator.free(layers);
        
        for (0..layers.len) |i| {
            layers[i] = MappedLayer.init(&offset, config, i);
        }
        
        const norm_weight = MappedTensor.init(offset, &[_]u32{config.dim});
        offset += config.dim / 2;
        
        if (offset > mapped_data.len) {
            return error.InsufficientMappedMemory;
        }
        
        return Parameters{
            .config = config,
            .mapped_data = mapped_data,
            .token_embedding = token_embedding,
            .layers = layers,
            .norm_weight = norm_weight,
        };
    }
    
    pub fn deinit(self: *Parameters) void {
        self.token_embedding.deinit();
        for (self.layers) |*l| l.deinit();
        std.heap.page_allocator.free(self.layers);
        self.norm_weight.deinit();
    }
    
    pub fn getTokenEmbedding(self: *Parameters, token: u32, dim_idx: u32) !f32 {
        if (token >= self.config.vocab_size) return error.TokenOutOfRange;
        if (dim_idx >= self.config.dim) return error.DimensionOutOfRange;
        return self.token_embedding.getSafe(self.mapped_data, &[_]u32{token, dim_idx});
    }
    
    pub fn getNormWeight(self: *Parameters, idx: u32) !f32 {
        if (idx >= self.config.dim) return error.DimensionOutOfRange;
        return self.norm_weight.getSafe(self.mapped_data, &[_]u32{idx});
    }
};

pub const MappedWeights = struct {
    data: []align(std.mem.page_size) u8,
    file: std.fs.File,
    allocator: std.mem.Allocator,
    
    pub fn deinit(self: *MappedWeights) void {
        std.posix.munmap(self.data);
        self.file.close();
    }
};

pub fn loadFromMappedFile(allocator: std.mem.Allocator, mapped: *MappedWeights) !Parameters {
    _ = allocator;
    
    const config = BritixConfig{
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
    
    try config.validate();
    const params = try Parameters.init(mapped.data, config);
    
    return params;
}
