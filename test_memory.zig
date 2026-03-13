// test_memory.zig - Memory leak detection for Britix
// "GPA is watching. No leaks shall pass!"

const std = @import("std");
const builtin = @import("builtin");

pub const MemoryTest = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    allocations: std.AutoHashMap(usize, AllocationInfo),
    peak_usage: usize = 0,
    current_usage: usize = 0,
    
    pub const AllocationInfo = struct {
        size: usize,
        // Stack trace removed - causes issues in tests
    };
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .allocations = std.AutoHashMap(usize, AllocationInfo).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        if (self.allocations.count() > 0) {
            std.debug.print("❌ Memory leaks detected!\n", .{});
            var iter = self.allocations.iterator();
            while (iter.next()) |entry| {
                std.debug.print("  Leaked {d} bytes at address 0x{x}\n", .{
                    entry.value_ptr.size, entry.key_ptr.*
                });
            }
        }
        self.allocations.deinit();
    }
    
    pub fn track_allocation(self: *Self, ptr: usize, size: usize) !void {
        try self.allocations.put(ptr, .{
            .size = size,
        });
        
        self.current_usage += size;
        self.peak_usage = @max(self.peak_usage, self.current_usage);
    }
    
    pub fn track_deallocation(self: *Self, ptr: usize) void {
        if (self.allocations.fetchRemove(ptr)) |kv| {
            self.current_usage -= kv.value.size;
        } else {
            std.debug.print("⚠️  Double free or invalid pointer: 0x{x}\n", .{ptr});
        }
    }
};

pub fn stress_test(iterations: usize) !void {
    std.debug.print("Running memory stress test for {d} iterations...\n", .{iterations});
    
    for (0..iterations) |i| {
        var gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
        defer _ = gpa.deinit();
        const alloc = gpa.allocator();
        
        // Allocate and free various sizes
        var ptrs = std.ArrayList([]u8).init(alloc);
        defer {
            for (ptrs.items) |p| alloc.free(p);
            ptrs.deinit();
        }
        
        // Random allocations
        var rng = std.Random.DefaultPrng.init(@intCast(i));
        const num_allocs = rng.random().intRangeLessThan(usize, 10, 100);
        
        for (0..num_allocs) |_| {
            const size = rng.random().intRangeLessThan(usize, 1, 1024 * 1024);
            const ptr = try alloc.alloc(u8, size);
            try ptrs.append(ptr);
        }
    }
}

test "memory - basic tracking" {
    var tracker = MemoryTest.init(std.testing.allocator);
    defer tracker.deinit();
    
    var gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    
    const ptr = try alloc.alloc(u8, 100);
    defer alloc.free(ptr);
    
    try tracker.track_allocation(@intFromPtr(ptr.ptr), 100);
    tracker.track_deallocation(@intFromPtr(ptr.ptr));
}
