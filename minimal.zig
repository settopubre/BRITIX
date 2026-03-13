// minimal.zig - Just open the file and do nothing
const std = @import("std");

pub fn main() !void {
    std.debug.print("1. Starting...\n", .{});
    
    var gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
    defer _ = gpa.deinit();
    _ = gpa.allocator();  // Just to satisfy, but not used
    
    std.debug.print("2. Opening file...\n", .{});
    const file = try std.fs.cwd().openFile("weights.bin", .{});
    defer file.close();
    
    std.debug.print("3. File opened\n", .{});
    
    const size = try file.getEndPos();
    std.debug.print("4. File size: {d:.2} GB\n", .{
        @as(f64, @floatFromInt(size)) / (1024*1024*1024)
    });
    
    std.debug.print("5. Sleeping for 10 seconds...\n", .{});
    std.time.sleep(10 * std.time.ns_per_s);
    
    std.debug.print("6. Done!\n", .{});
}
