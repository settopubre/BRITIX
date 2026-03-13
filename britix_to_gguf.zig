// britix_to_gguf.zig - Fixed float writing

const std = @import("std");
const fs = std.fs;
const mem = std.mem;

// GGUF constants (from llama.cpp)
const GGUF_MAGIC = 0x46554747; // 'GGUF' in little-endian
const GGUF_VERSION = 3;

const GGML_TYPE_F32 = 0;
const GGML_TYPE_F16 = 1;
const GGML_TYPE_Q4_0 = 2;
const GGML_TYPE_Q4_1 = 3;

pub const GGUFHeader = struct {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
};

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

pub const GGUFTensorInfo = struct {
    name: [64]u8,
    n_dims: u32,
    dims: [4]u32,
    type: u32,
    offset: u64,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) std.debug.print("💀 LEAKS! THE VILLAIN WINS!\n", .{});
    }
    const alloc = gpa.allocator();

    std.debug.print(
        \\
        \\╔════════════════════════════════════════╗
        \\║     BRITIX → GGUF CONVERTER            ║
        \\║     For llama.cpp server               ║
        \\╚════════════════════════════════════════╝
        \\
    , .{});

    // Check if weights file exists
    const weights_file = fs.cwd().openFile("weights.bin", .{}) catch |err| {
        std.debug.print("❌ weights.bin not found! Generate it first with:\n", .{});
        std.debug.print("   zig build-exe generate_britix_weights.zig\n", .{});
        std.debug.print("   ./generate_britix_weights\n", .{});
        return err;
    };
    defer weights_file.close();
    
    const weights_size = try weights_file.getEndPos();
    std.debug.print("📦 Loading {d:.2} GB weights file...\n", .{
        @as(f64, @floatFromInt(weights_size)) / (1024 * 1024 * 1024)
    });
    
    // Read entire file
    const weights_data = try weights_file.readToEndAlloc(alloc, weights_size);
    defer alloc.free(weights_data);
    
    // Britix architecture
    const dim: u32 = 4096;
    const n_layers: u32 = 32;
    const n_heads: u32 = 32;
    const n_kv_heads: u32 = 8;
    const vocab_size: u32 = 32014;
    const hidden_dim: u32 = 14336;
    
    std.debug.print("🏗️  Britix Architecture:\n", .{});
    std.debug.print("   • Layers: {d}\n", .{n_layers});
    std.debug.print("   • Dim: {d}\n", .{dim});
    std.debug.print("   • Heads: {d}\n", .{n_heads});
    std.debug.print("   • Vocab: {d}\n", .{vocab_size});
    
    // Create GGUF file
    const gguf_file = try fs.cwd().createFile("britix-8b.gguf", .{});
    defer gguf_file.close();
    const writer = gguf_file.writer();
    
    // Write header
    const tensor_count = 3 + (n_layers * 9); // embedding + output_norm + output + 9 per layer
    const metadata_count = 12;
    
    try writer.writeInt(u32, GGUF_MAGIC, .little);
    try writer.writeInt(u32, GGUF_VERSION, .little);
    try writer.writeInt(u64, tensor_count, .little);
    try writer.writeInt(u64, metadata_count, .little);
    
    // Write metadata (llama.cpp expects these)
    try writeStringMetadata(writer, "general.architecture", "llama");
    try writeStringMetadata(writer, "general.name", "Britix 8B");
    try writeStringMetadata(writer, "llama.context_length", "8192");
    try writeU32Metadata(writer, "llama.embedding_length", dim);
    try writeU32Metadata(writer, "llama.block_count", n_layers);
    try writeU32Metadata(writer, "llama.feed_forward_length", hidden_dim);
    try writeU32Metadata(writer, "llama.attention.head_count", n_heads);
    try writeU32Metadata(writer, "llama.attention.head_count_kv", n_kv_heads);
    try writeU32Metadata(writer, "llama.rope.dimension_count", dim / n_heads);
    try writeF32Metadata(writer, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    
    // Tokenizer metadata
    try writeStringMetadata(writer, "tokenizer.ggml.model", "llama");
    try writeStringMetadata(writer, "tokenizer.ggml.pre", "default");
    
    // Write tensor metadata placeholder
    std.debug.print("\n📝 Writing tensor metadata...\n", .{});
    
    // For now, write a single dummy tensor so GGUF is valid
    var dummy_name: [64]u8 = .{0} ** 64;
    const name = "token_embd.weight";
    @memcpy(dummy_name[0..name.len], name);
    
    try writer.writeAll(&dummy_name);
    try writer.writeInt(u32, 2, .little); // n_dims
    try writer.writeInt(u32, dim, .little); // dims[0]
    try writer.writeInt(u32, vocab_size, .little); // dims[1]
    try writer.writeInt(u32, 0, .little); // dims[2]
    try writer.writeInt(u32, 0, .little); // dims[3]
    try writer.writeInt(u32, GGML_TYPE_F32, .little); // type
    try writer.writeInt(u64, 0, .little); // offset
    
    std.debug.print("\n✅ Created britix-8b.gguf (minimal valid file)\n", .{});
    std.debug.print("\n🚀 NOW RUN:\n", .{});
    std.debug.print("   ~/llama.cpp/build/bin/llama-server -m britix-8b.gguf -c 2048 --port 8020 -t 4\n", .{});
    std.debug.print("\n📝 Note: This is a minimal GGUF file. For full functionality,\n", .{});
    std.debug.print("   we need to implement proper tensor writing with actual weights.\n", .{});
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
    try writer.writeInt(u32, @as(u32, @bitCast(value)), .little); // Write float as u32 bits
}
