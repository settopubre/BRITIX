// queue.zig - Michael-Scott lock-free queue
// "Lock-free, worry-free, jolly good!"

const std = @import("std");
const Atomic = std.atomic.Value;

pub fn Queue(comptime T: type) type {
    return struct {
        const Self = @This();
        
        const Node = struct {
            data: T,
            next: Atomic(?*Node) = Atomic(?*Node).init(null),
        };
        
        head: Atomic(?*Node),
        tail: Atomic(?*Node),
        allocator: std.mem.Allocator,
        
        pub fn init(allocator: std.mem.Allocator) !*Self {
            // Create dummy node
            const dummy = try allocator.create(Node);
            dummy.* = Node{
                .data = undefined,
                .next = Atomic(?*Node).init(null),
            };
            
            const self = try allocator.create(Self);
            self.* = Self{
                .head = Atomic(?*Node).init(dummy),
                .tail = Atomic(?*Node).init(dummy),
                .allocator = allocator,
            };
            return self;
        }
        
        pub fn deinit(self: *Self) void {
            // Pop all nodes
            while (self.pop()) |_| {}
            
            // Free dummy node
            const dummy = self.head.raw;
            self.allocator.destroy(dummy.?);
            self.allocator.destroy(self);
        }
        
        pub fn push(self: *Self, data: T) !void {
            const node = try self.allocator.create(Node);
            node.* = Node{
                .data = data,
                .next = Atomic(?*Node).init(null),
            };
            
            while (true) {
                const tail = self.tail.raw;
                const next = tail.?.next.raw;
                
                if (tail == self.tail.raw) {
                    if (next == null) {
                        // Try to link node at tail
                        if (@cmpxchgWeak(?*Node, &tail.?.next.raw, null, node, .release, .monotonic)) |_| {
                            continue;
                        }
                        // Try to swing tail to node
                        _ = @cmpxchgWeak(?*Node, &self.tail.raw, tail, node, .release, .monotonic);
                        break;
                    } else {
                        // Tail is falling behind, help it
                        _ = @cmpxchgWeak(?*Node, &self.tail.raw, tail, next, .release, .monotonic);
                    }
                }
            }
        }
        
        pub fn pop(self: *Self) ?T {
            while (true) {
                const head = self.head.raw;
                const tail = self.tail.raw;
                const next = head.?.next.raw;
                
                if (head == self.head.raw) {
                    if (head == tail) {
                        if (next == null) return null;
                        // Tail is falling behind
                        _ = @cmpxchgWeak(?*Node, &self.tail.raw, tail, next, .release, .monotonic);
                    } else {
                        const data = next.?.data;
                        if (@cmpxchgWeak(?*Node, &self.head.raw, head, next, .release, .monotonic)) |_| {
                            continue;
                        }
                        self.allocator.destroy(head.?);
                        return data;
                    }
                }
            }
        }
        
        pub fn isEmpty(self: *Self) bool {
            const head = self.head.raw;
            const next = head.?.next.raw;
            return next == null;
        }
    };
}

test "queue - single producer consumer" {
    const allocator = std.testing.allocator;
    var queue = try Queue(usize).init(allocator);
    defer queue.deinit();
    
    try queue.push(42);
    try queue.push(43);
    
    try std.testing.expectEqual(queue.pop(), 42);
    try std.testing.expectEqual(queue.pop(), 43);
    try std.testing.expectEqual(queue.pop(), null);
}

test "queue - concurrent" {
    const allocator = std.testing.allocator;
    var queue = try Queue(usize).init(allocator);
    defer queue.deinit();
    
    const num_items = 1000;
    var producer_done = false;
    
    // Producer thread
    const producer = try std.Thread.spawn(.{}, struct {
        fn run(q: *Queue(usize), n: usize, done: *bool) void {
            for (0..n) |i| {
                q.push(i) catch unreachable;
            }
            done.* = true;
        }
    }.run, .{ queue, num_items, &producer_done });
    
    // Consumer
    var count: usize = 0;
    while (count < num_items) {
        if (queue.pop()) |_| {
            count += 1;
        }
    }
    
    producer.join();
    try std.testing.expectEqual(count, num_items);
}
