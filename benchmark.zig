// benchmark.zig - Benchmark Britix performance
// "Measuring speed with a stopwatch and a stiff upper lip!"

const std = @import("std");
const builtin = @import("builtin");

pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    total_time_ns: u64,
    ops_per_second: f64,
    
    pub fn format(self: BenchmarkResult) void {
        std.debug.print("  {s:30} {d:12} ops/s\n", .{
            self.name,
            @as(u64, @intFromFloat(self.ops_per_second)),
        });
    }
};

pub fn benchmark(comptime name: []const u8, iterations: usize, func: anytype) !BenchmarkResult {
    const start = std.time.nanoTimestamp();
    
    for (0..iterations) |i| {
        @call(.auto, func, .{i});
    }
    
    const end = std.time.nanoTimestamp();
    const total_time = @as(u64, @intCast(end - start));
    
    return BenchmarkResult{
        .name = name,
        .iterations = iterations,
        .total_time_ns = total_time,
        .ops_per_second = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0),
    };
}

fn mathBenchmark(i: usize) void {
    var x: f32 = @as(f32, @floatFromInt(i & 0xFF)) / 255.0;
    x = @exp(x);
    x = @log(x + 1.0);
    x = @sqrt(x);
    std.mem.doNotOptimizeAway(x);
}

fn matrixMulBenchmark(i: usize) void {
    const n = 64;
    var a: [n][n]f32 = undefined;
    var b: [n][n]f32 = undefined;
    var c: [n][n]f32 = undefined;
    
    // Initialize
    for (0..n) |j| {
        for (0..n) |k| {
            a[j][k] = @as(f32, @floatFromInt((j * n + k + i) & 0xFF)) / 255.0;
            b[j][k] = @as(f32, @floatFromInt((j * n + k + i * 2) & 0xFF)) / 255.0;
        }
    }
    
    // Multiply
    for (0..n) |j| {
        for (0..n) |k| {
            var sum: f32 = 0.0;
            for (0..n) |l| {
                sum += a[j][l] * b[l][k];
            }
            c[j][k] = sum;
        }
    }
    
    std.mem.doNotOptimizeAway(&c);
}

pub fn main() !void {
    std.debug.print("\n", .{});
    std.debug.print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", .{});
    std.debug.print("┃                    BRITIX BENCHMARK SUITE                          ┃\n", .{});
    std.debug.print("┃              Measuring speed with British precision                ┃\n", .{});
    std.debug.print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n", .{});
    
    std.debug.print("\nSystem Info:\n", .{});
    std.debug.print("  CPU: {s}\n", .{builtin.cpu.arch});
    std.debug.print("  Cores: {}\n", .{std.Thread.getCpuCount() catch 16});
    
    var results = std.ArrayList(BenchmarkResult).init(std.heap.page_allocator);
    defer results.deinit();
    
    std.debug.print("\nRunning benchmarks...\n", .{});
    
    // Math operations
    try results.append(try benchmark("Math ops (exp/log/sqrt)", 1_000_000, mathBenchmark));
    
    // Matrix multiplication
    try results.append(try benchmark("64x64 matrix multiply", 1000, matrixMulBenchmark));
    
    // Memory bandwidth
    // TODO: Add memory benchmark
    
    std.debug.print("\nResults:\n", .{});
    var total_ops: f64 = 0;
    for (results.items) |r| {
        r.format();
        total_ops += r.ops_per_second;
    }
    
    std.debug.print("\n🇬🇧 Total performance: {d:.2} million ops/s\n", .{total_ops / 1_000_000.0});
    
    if (total_ops > 100_000_000) {
        std.debug.print("   Jolly fast! The samurai sword is sharp!\n", .{});
    } else {
        std.debug.print("   Not bad, but we can do better. Time for optimisation!\n", .{});
    }
}
