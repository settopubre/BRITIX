// profiler.zig - Performance profiling for Britix
// "Finding bottlenecks with Sherlock Holmes precision."

const std = @import("std");
const builtin = @import("builtin");
const time = std.time;

pub const ProfileScope = struct {
    const Self = @This();
    
    name: []const u8,
    start: i128,
    profiler: *Profiler,
    
    pub fn init(profiler: *Profiler, name: []const u8) ProfileScope {
        return ProfileScope{
            .name = name,
            .start = time.nanoTimestamp(),
            .profiler = profiler,
        };
    }
    
    pub fn deinit(self: *ProfileScope) void {
        const end = time.nanoTimestamp();
        const duration = end - self.start;
        self.profiler.record(self.name, @intCast(duration));
    }
};

pub const ProfileRecord = struct {
    name: []const u8,
    total_time: u64,
    call_count: usize,
    min_time: u64,
    max_time: u64,
};

pub const Profiler = struct {
    const Self = @This();
    
    records: std.StringHashMap(ProfileRecord),
    enabled: bool,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .records = std.StringHashMap(ProfileRecord).init(allocator),
            .enabled = true,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        var iter = self.records.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.records.deinit();
    }
    
    pub fn record(self: *Self, name: []const u8, duration_ns: u64) void {
        if (!self.enabled) return;
        
        var entry = self.records.getOrPut(name) catch return;
        if (!entry.found_existing) {
            // First time seeing this name - make a copy
            const name_copy = self.allocator.dupe(u8, name) catch return;
            entry.key_ptr.* = name_copy;
            entry.value_ptr.* = ProfileRecord{
                .name = name_copy,
                .total_time = duration_ns,
                .call_count = 1,
                .min_time = duration_ns,
                .max_time = duration_ns,
            };
        } else {
            // Update existing record
            entry.value_ptr.total_time += duration_ns;
            entry.value_ptr.call_count += 1;
            entry.value_ptr.min_time = @min(entry.value_ptr.min_time, duration_ns);
            entry.value_ptr.max_time = @max(entry.value_ptr.max_time, duration_ns);
        }
    }
    
    pub fn scope(self: *Self, name: []const u8) ProfileScope {
        return ProfileScope.init(self, name);
    }
    
    pub fn reset(self: *Self) void {
        var iter = self.records.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.records.clearRetainingCapacity();
    }
    
    pub fn print_report(self: *Self) void {
        std.debug.print("\n", .{});
        std.debug.print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", .{});
        std.debug.print("┃                         PROFILE REPORT                             ┃\n", .{});
        std.debug.print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n", .{});
        std.debug.print("\n", .{});
        
        var iter = self.records.iterator();
        while (iter.next()) |entry| {
            const r = entry.value_ptr;
            std.debug.print("  {s}: calls={d}, total={d}ns\n", .{r.name, r.call_count, r.total_time});
        }
    }
};

test "profiler - basic recording" {
    const allocator = std.testing.allocator;
    var profiler = Profiler.init(allocator);
    defer profiler.deinit();
    
    profiler.record("test", 1000);
    profiler.record("test", 500);
    
    const record = profiler.records.get("test").?;
    try std.testing.expectEqual(@as(usize, 2), record.call_count);
    try std.testing.expectEqual(@as(u64, 1500), record.total_time);
}

test "profiler - scope" {
    const allocator = std.testing.allocator;
    var profiler = Profiler.init(allocator);
    defer profiler.deinit();
    
    {
        var scope = profiler.scope("test_scope");
        defer scope.deinit();
        // Do nothing
    }
    
    // Just test that it doesn't crash
    try std.testing.expect(true);
}
