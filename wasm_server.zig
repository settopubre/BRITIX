// wasm_server.zig - Britix WebAssembly module
const std = @import("std");
const britix = @import("britix.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

// Simple response generator
fn getResponse(input: []const u8) []const u8 {
    if (std.mem.indexOf(u8, input, "hello") != null) {
        return "Good day, Governor! How may I assist?";
    }
    if (std.mem.indexOf(u8, input, "tea") != null) {
        return "Tea time is at 3pm sharp, Governor. Earl Grey, no sugar.";
    }
    if (std.mem.indexOf(u8, input, "1+1") != null) {
        return "2, Governor. Even a Britix knows basic arithmetic.";
    }
    if (std.mem.indexOf(u8, input, "how are you") != null) {
        return "Splendid, Governor! Ready to serve.";
    }
    return "I'm here to assist you, Governor.";
}

export fn init_britix() i32 {
    return 1;
}

export fn generate_text(input_ptr: [*]const u8, input_len: usize, output_ptr: [*]u8, output_len: usize) i32 {
    _ = output_len;
    const input = input_ptr[0..input_len];
    
    const response = getResponse(input);
    
    @memcpy(output_ptr[0..response.len], response);
    return @intCast(response.len);
}

export fn alloc(size: usize) ?[*]u8 {
    const buffer = allocator.alloc(u8, size) catch return null;
    return buffer.ptr;
}

export fn free(ptr: ?[*]u8, size: usize) void {
    if (ptr) |p| {
        const slice = p[0..size];
        allocator.free(slice);
    }
}

pub fn main() void {}
