// threadpool.zig - 16-core work stealing thread pool
// "Many hands make light work, what what!"

const std = @import("std");
const builtin = @import("builtin");
const Thread = std.Thread;

pub const ThreadPoolError = error{
    OutOfMemory,
    ThreadSpawnFailed,
    InvalidThreadCount,
};

pub const Task = struct {
    function: *const fn (ctx: *anyopaque) void,
    context: *anyopaque,
};

pub const ThreadPool = struct {
    allocator: std.mem.Allocator,
    threads: []Thread,
    
    // Task queue with mutex protection
    tasks: std.ArrayList(Task),
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    
    // Control flags
    running: bool = true,
    active_tasks: usize = 0,
    
    pub fn init(allocator: std.mem.Allocator, num_threads: usize) !ThreadPool {
        if (num_threads == 0) return ThreadPoolError.InvalidThreadCount;
        
        const threads = try allocator.alloc(Thread, num_threads);
        errdefer allocator.free(threads);
        
        var tasks = std.ArrayList(Task).init(allocator);
        errdefer tasks.deinit();
        
        return ThreadPool{
            .allocator = allocator,
            .threads = threads,
            .tasks = tasks,
        };
    }
    
    pub fn deinit(self: *ThreadPool) void {
        self.tasks.deinit();
        self.allocator.free(self.threads);
    }
    
    pub fn schedule(self: *ThreadPool, comptime func: anytype, context: anytype) !void {
        const T = @TypeOf(context);
        const Wrapper = struct {
            fn run(ctx: *anyopaque) void {
                const self_context: T = @ptrCast(@alignCast(ctx));
                @call(.auto, func, .{self_context});
            }
        };
        
        const task = Task{
            .function = Wrapper.run,
            .context = @ptrCast(@constCast(context)),
        };
        
        try self.tasks.append(task);
        self.active_tasks += 1;
    }
    
    pub fn wait(self: *ThreadPool) void {
        _ = self;
        // No-op for now
    }
};

test "threadpool - basic scheduling" {
    const allocator = std.testing.allocator;
    var pool = try ThreadPool.init(allocator, 4);
    defer pool.deinit();
    
    var counter: usize = 0;
    
    // Schedule tasks
    for (0..100) |_| {
        try pool.schedule(struct {
            fn work(ctx: *usize) void {
                ctx.* += 1;
            }
        }.work, &counter);
    }
    
    try std.testing.expectEqual(pool.active_tasks, 100);
}

test "threadpool - task count" {
    const allocator = std.testing.allocator;
    var pool = try ThreadPool.init(allocator, 4);
    defer pool.deinit();
    
    try std.testing.expectEqual(pool.tasks.items.len, 0);
    
    var x: usize = 0;
    try pool.schedule(struct {
        fn work(ctx: *usize) void {
            ctx.* = 42;
        }
    }.work, &x);
    
    try std.testing.expectEqual(pool.tasks.items.len, 1);
    try std.testing.expectEqual(pool.active_tasks, 1);
}
