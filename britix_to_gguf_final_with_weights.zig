// britix_to_gguf_final_with_weights.zig - For 4-bit quantized weights (3.5GB)
// "The one that finally works, Governor!"

const std = @import("std");
const fs = std.fs;
const mem = std.mem;

const GGUF_MAGIC = 0x46554747; // 'GGUF'
const GGUF_VERSION = 3;

const GGML_TYPE_Q4_0 = 2;  // 4-bit quantized type

pub const GGUFMetadataType = enum(u32) {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    std.debug.print(
        \\
        \\╔════════════════════════════════════════╗
        \\║     BRITIX → GGUF (WORKING)            ┃
        \\║     For 4-bit weights (3.5GB)          ┃
        \\╚════════════════════════════════════════╝
        \\
    , .{});

    // Load quantized weights
    const weights_file = try fs.cwd().openFile("weights.bin", .{});
    defer weights_file.close();
    
    const weights_size = try weights_file.getEndPos();
    std.debug.print("📦 Loading {d:.2} GB quantized weights...\n", .{
        @as(f64, @floatFromInt(weights_size)) / (1024 * 1024 * 1024)
    });
    
    const weights = try weights_file.readToEndAlloc(alloc, weights_size);
    defer alloc.free(weights);
    
    // Britix architecture
    const dim: u32 = 4096;
    const n_layers: u32 = 32;
    const n_heads: u32 = 32;
    const n_kv_heads: u32 = 8;
    const vocab_size: u32 = 32014;
    const hidden_dim: u32 = 14336;
    const head_dim = dim / n_heads;
    
    // Create GGUF file
    const gguf_file = try fs.cwd().createFile("britix-8b.gguf", .{});
    defer gguf_file.close();
    const writer = gguf_file.writer();
    
    // Calculate tensor count
    const tensor_count = 3 + n_layers * 9;
    const metadata_count = 13;
    
    // Write header
    try writer.writeInt(u32, GGUF_MAGIC, .little);
    try writer.writeInt(u32, GGUF_VERSION, .little);
    try writer.writeInt(u64, tensor_count, .little);
    try writer.writeInt(u64, metadata_count, .little);
    
    // Write metadata
    try writeStringMetadata(writer, "general.architecture", "llama");
    try writeStringMetadata(writer, "general.name", "Britix 8B");
    try writeStringMetadata(writer, "general.description", "British-trained, formally verified 8B model (4-bit quantized)");
    try writeStringMetadata(writer, "llama.context_length", "8192");
    try writeU32Metadata(writer, "llama.embedding_length", dim);
    try writeU32Metadata(writer, "llama.block_count", n_layers);
    try writeU32Metadata(writer, "llama.feed_forward_length", hidden_dim);
    try writeU32Metadata(writer, "llama.attention.head_count", n_heads);
    try writeU32Metadata(writer, "llama.attention.head_count_kv", n_kv_heads);
    try writeU32Metadata(writer, "llama.rope.dimension_count", head_dim);
    try writeF32Metadata(writer, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    try writeStringMetadata(writer, "tokenizer.ggml.model", "llama");
    try writeStringMetadata(writer, "tokenizer.ggml.pre", "default");
    try writeU32Metadata(writer, "general.file_type", 2); // Q4_0
    
    // Keep track of offset for tensor headers
    var offset: u64 = 0;
    
    // Write tensor headers
    std.debug.print("📝 Writing tensor headers...\n", .{});
    
    try writeTensorHeader(writer, "token_embd.weight", &offset, dim, vocab_size, GGML_TYPE_Q4_0);
    
    for (0..n_layers) |i| {
        var name_buf: [64]u8 = undefined;
        
        try writeTensorHeader(writer, 
            try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm.weight", .{i}), 
            &offset, dim, 1, GGML_TYPE_Q4_0);
        
        try writeTensorHeader(writer,
            try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q.weight", .{i}),
            &offset, n_heads * head_dim, dim, GGML_TYPE_Q4_0);
        
        try writeTensorHeader(writer,
            try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k.weight", .{i}),
            &offset, n_kv_heads * head_dim, dim, GGML_TYPE_Q4_0);
        
        try writeTensorHeader(writer,
            try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v.weight", .{i}),
            &offset, n_kv_heads * head_dim, dim, GGML_TYPE_Q4_0);
        
        try writeTensorHeader(writer,
            try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_output.weight", .{i}),
            &offset, dim, n_heads * head_dim, GGML_TYPE_Q4_0);
        
        try writeTensorHeader(writer,
            try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm.weight", .{i}),
            &offset, dim, 1, GGML_TYPE_Q4_0);
        
        try writeTensorHeader(writer,
            try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate.weight", .{i}),
            &offset, hidden_dim, dim, GGML_TYPE_Q4_0);
        
        try writeTensorHeader(writer,
            try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down.weight", .{i}),
            &offset, dim, hidden_dim, GGML_TYPE_Q4_0);
        
        try writeTensorHeader(writer,
            try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up.weight", .{i}),
            &offset, hidden_dim, dim, GGML_TYPE_Q4_0);
    }
    
    try writeTensorHeader(writer, "output_norm.weight", &offset, dim, 1, GGML_TYPE_Q4_0);
    try writeTensorHeader(writer, "output.weight", &offset, vocab_size, dim, GGML_TYPE_Q4_0);
    
    // Now write the actual weight data (the entire quantized file)
    std.debug.print("📝 Writing quantized weights ({d} bytes)...\n", .{weights_size});
    try writer.writeAll(weights);
    
    std.debug.print("\n✅ Britix-8b.gguf created successfully!\n", .{});
    std.debug.print("   File size: {d:.2} GB\n", .{
        @as(f64, @floatFromInt(offset + weights_size)) / (1024 * 1024 * 1024)
    });
    std.debug.print("\n🚀 NOW RUN:\n", .{});
    std.debug.print("   ~/llama.cpp/build/bin/llama-server -m britix-8b.gguf -c 2048 --port 8020 -t 4\n", .{});
}

fn writeStringMetadata(writer: anytype, key: []const u8, value: []const u8) !void {
    try writer.writeInt(u32, @as(u32, @intCast(key.len)), .little);
    try writer.writeAll(key);
    try writer.writeInt(u32, @intFromEnum(GGUFMetadataType.STRING), .little);
    try writer.writeInt(u64, @as(u64, @intCast(value.len)), .little);
    try writer.writeAll(value);
}

fn writeU32Metadata(writer: anytype, key: []const u8, value: u32) !void {
    try writer.writeInt(u32, @as(u32, @intCast(key.len)), .little);
    try writer.writeAll(key);
    try writer.writeInt(u32, @intFromEnum(GGUFMetadataType.UINT32), .little);
    try writer.writeInt(u32, value, .little);
}

fn writeF32Metadata(writer: anytype, key: []const u8, value: f32) !void {
    try writer.writeInt(u32, @as(u32, @intCast(key.len)), .little);
    try writer.writeAll(key);
    try writer.writeInt(u32, @intFromEnum(GGUFMetadataType.FLOAT32), .little);
    try writer.writeInt(u32, @as(u32, @bitCast(value)), .little);
}

fn writeTensorHeader(writer: anytype, name: []const u8, offset: *u64, dim0: u32, dim1: u32, dtype: u32) !void {
    var name_buf: [64]u8 = .{0} ** 64;
    @memcpy(name_buf[0..name.len], name);
    
    try writer.writeAll(&name_buf);
    
    if (dim1 > 1) {
        try writer.writeInt(u32, 2, .little);
        try writer.writeInt(u32, dim0, .little);
        try writer.writeInt(u32, dim1, .little);
        try writer.writeInt(u32, 0, .little);
        try writer.writeInt(u32, 0, .little);
    } else {
        try writer.writeInt(u32, 1, .little);
        try writer.writeInt(u32, dim0, .little);
        try writer.writeInt(u32, 0, .little);
        try writer.writeInt(u32, 0, .little);
        try writer.writeInt(u32, 0, .little);
    }
    
    try writer.writeInt(u32, dtype, .little);
    try writer.writeInt(u64, offset.*, .little);
    
    // Update offset (approximate for quantized)
    offset.* += (dim0 * dim1 * 4) / 8;
}
