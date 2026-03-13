// run.zig - Quick test runner for Britix
// "Running tests with proper British precision, what!"

const std = @import("std");
const builtin = @import("builtin");

pub fn main() !void {
    std.debug.print("🧪 Running Britix tests...\n", .{});
    
    const modules = [_][]const u8{
        "parameters",
        "quantize",
        "cache",
        "threadpool",
        "queue",
        "barrier",
        "math",
        "random",
        "logger",
        "errors",
    };
    
    var passed: usize = 0;
    var failed: usize = 0;
    
    for (modules) |module| {
        std.debug.print("  Testing {s}... ", .{module});
        
        const result = try std.process.Child.run(.{
            .allocator = std.heap.page_allocator,
            .argv = &[_][]const u8{ "zig", "test", std.fmt.allocPrint(std.heap.page_allocator, "{s}.zig", .{module}) catch unreachable },
        });
        
        if (result.term.Exited == 0) {
            std.debug.print("✅ Passed\n", .{});
            passed += 1;
        } else {
            std.debug.print("❌ Failed\n", .{});
            std.debug.print("{s}\n", .{result.stderr});
            failed += 1;
        }
    }
    
    std.debug.print("\n📊 Results: {d} passed, {d} failed\n", .{ passed, failed });
    
    if (failed == 0) {
        std.debug.print("🇬🇧 Jolly good! All tests pass, what what!\n", .{});
    } else {
        std.debug.print("😱 Oh dear, {d} tests failed. Time for tea and debugging.\n", .{failed});
    }
}
