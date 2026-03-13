// test_speed.zig - Performance testing for Britix
// "Measuring speed with a stopwatch and a stiff upper lip!"

const std = @import("std");
const builtin = @import("builtin");

const parameters = @import("parameters.zig");
const tokenizer = @import("tokenizer.zig");
const threadpool = @import("threadpool.zig");

pub const SpeedTestConfig = struct {
    num_tokens: usize = 100,
    num_warmup: usize = 10,
    batch_sizes: []const usize = &.{ 1, 2, 4, 8 },
    sequence_lengths: []const usize = &.{ 128, 256, 512, 1024 },
};

pub const SpeedTestResult = struct {
    name: []const u8,
    tokens_per_second: f64,
    memory_bandwidth_gb_s: f64,
    flops: f64,
};

pub fn test_memory_bandwidth(allocator: std.mem.Allocator, size_mb: usize) !f64 {
    const num_elements = size_mb * 1024 * 1024 / @sizeOf(f32);
    
    const a = try allocator.alloc(f32, num_elements);
    defer allocator.free(a);
    
    const b = try allocator.alloc(f32, num_elements);
    defer allocator.free(b);
    
    const c = try allocator.alloc(f32, num_elements);
    defer allocator.free(c);
    
    // Initialize
    for (0..num_elements) |i| {
        a[i] = @as(f32, @floatFromInt(i & 0xFF));
        b[i] = @as(f32, @floatFromInt((i * 2) & 0xFF));
    }
    
    // Measure bandwidth: C = A + B (reads A,B; writes C)
    const start = std.time.nanoTimestamp();
    
    for (0..num_elements) |i| {
        c[i] = a[i] + b[i];
    }
    
    const end = std.time.nanoTimestamp();
    const elapsed_s = @as(f64, @floatFromInt(end - start)) / 1_000_000_000.0;
    
    // Bytes transferred: read a (size) + read b (size) + write c (size)
    const bytes_transferred = 3 * size_mb * 1024 * 1024;
    const bandwidth = @as(f64, @floatFromInt(bytes_transferred)) / elapsed_s / 1_000_000_000.0;
    
    return bandwidth;
}

pub fn print_results(results: []const SpeedTestResult) void {
    std.debug.print("\n", .{});
    std.debug.print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", .{});
    std.debug.print("┃              BRITIX SPEED TEST RESULTS               ┃\n", .{});
    std.debug.print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n", .{});
    std.debug.print("\n", .{});
    
    for (results) |r| {
        std.debug.print("  {s:30} {d:8.2} tokens/s\n", .{r.name, r.tokens_per_second});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    
    std.debug.print("⚡ Running Britix speed tests...\n", .{});
    
    // Memory bandwidth test
    const bandwidth = try test_memory_bandwidth(alloc, 100);
    std.debug.print("Memory bandwidth: {d:.2} GB/s\n", .{bandwidth});
}

test "speed - basic test" {
    const allocator = std.testing.allocator;
    _ = try test_memory_bandwidth(allocator, 1);
}
