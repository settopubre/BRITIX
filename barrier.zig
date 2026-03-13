// barrier.zig - Simple barrier for thread synchronization
// "All together now! Like a proper British chorus!"

const std = @import("std");
const Atomic = std.atomic.Value;

pub const Barrier = struct {
    const Self = @This();
    
    count: usize,
    arrived: Atomic(usize) = Atomic(usize).init(0),
    
    pub fn init(count: usize) Barrier {
        return Barrier{
            .count = count,
        };
    }
    
    pub fn wait(self: *Self) void {
        _ = self.arrived.fetchAdd(1, .monotonic);
        
        // Spin until all threads have arrived
        while (self.arrived.load(.acquire) < self.count) {
            std.atomic.spinLoopHint();
        }
    }
    
    pub fn reset(self: *Self) void {
        self.arrived.store(0, .release);
    }
};

test "barrier - basic synchronization" {
    const allocator = std.testing.allocator;
    var barrier = Barrier.init(4);
    
    var threads = std.ArrayList(std.Thread).init(allocator);
    defer threads.deinit();
    
    var counter: usize = 0;
    
    for (0..4) |i| {
        const thread = try std.Thread.spawn(.{}, struct {
            fn run(b: *Barrier, c: *usize, id: usize) void {
                _ = id;
                // Increment counter
                _ = @atomicRmw(usize, c, .Add, 1, .monotonic);
                // Wait at barrier
                b.wait();
            }
        }.run, .{ &barrier, &counter, i });
        try threads.append(thread);
    }
    
    // Wait for threads with timeout
    for (threads.items) |t| {
        t.join();
    }
    
    barrier.reset();
    try std.testing.expectEqual(counter, 4);
}
