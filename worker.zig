// worker.zig - Worker threads for Britix thread pool
// "Many hands make light work. Jolly good teamwork!"

const std = @import("std");
const builtin = @import("builtin");

const threadpool = @import("threadpool.zig");

pub const Worker = struct {
    const Self = @This();
    
    id: usize,
    pool: *threadpool.ThreadPool,
    tasks_processed: usize = 0,
    
    pub fn init(pool: *threadpool.ThreadPool, id: usize) Worker {
        return Worker{
            .id = id,
            .pool = pool,
        };
    }
    
    pub fn getStats(self: *Self) struct { processed: usize } {
        return .{
            .processed = self.tasks_processed,
        };
    }
};

pub const WorkerPool = struct {
    const Self = @This();
    
    workers: []Worker,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, pool: *threadpool.ThreadPool, num_workers: usize) !WorkerPool {
        var workers = try allocator.alloc(Worker, num_workers);
        errdefer allocator.free(workers);
        
        for (0..num_workers) |i| {
            workers[i] = Worker.init(pool, i);
        }
        
        return WorkerPool{
            .workers = workers,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.workers);
    }
    
    pub fn getTotalStats(self: *Self) struct { processed: usize } {
        var total_processed: usize = 0;
        
        for (self.workers) |*worker| {
            const stats = worker.getStats();
            total_processed += stats.processed;
        }
        
        return .{
            .processed = total_processed,
        };
    }
};

test "worker - basic functionality" {
    const allocator = std.testing.allocator;
    
    // Create thread pool
    var pool = try threadpool.ThreadPool.init(allocator, 2);
    defer pool.deinit();
    
    // Create worker pool (no actual threads spawned)
    var workers = try WorkerPool.init(allocator, &pool, 2);
    defer workers.deinit();
    
    try std.testing.expect(workers.workers.len == 2);
}
