// logger.zig - British gentleman's logger
// "I say, logging something rather important, what!"

const std = @import("std");
const builtin = @import("builtin");

pub const LogLevel = enum(u3) {
    debug = 0,
    info = 1,
    warn = 2,
    err = 3,
    fatal = 4,
    
    pub fn prefix(self: LogLevel) []const u8 {
        return switch (self) {
            .debug => "🔍",
            .info => "💬",
            .warn => "⚠️",
            .err => "❌",
            .fatal => "💀",
        };
    }
    
    pub fn britishPhrase(self: LogLevel) []const u8 {
        return switch (self) {
            .debug => "I say, rather interesting...",
            .info => "Jolly good!",
            .warn => "Oh dear, that's not quite right...",
            .err => "Blast! An error!",
            .fatal => "Well, this is a fine kettle of fish!",
        };
    }
};

pub const Logger = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    level: LogLevel,
    start_time: i64,
    file: ?std.fs.File,
    
    pub fn init(allocator: std.mem.Allocator, level: LogLevel, log_file: ?[]const u8) !Self {
        const file = if (log_file) |path|
            try std.fs.cwd().createFile(path, .{})
        else
            null;
        
        return Self{
            .allocator = allocator,
            .level = level,
            .start_time = std.time.milliTimestamp(),
            .file = file,
        };
    }
    
    pub fn deinit(self: *Self) void {
        if (self.file) |*f| {
            f.close();
        }
    }
    
    pub fn log(self: *Self, level: LogLevel, comptime format: []const u8, args: anytype) !void {
        if (@intFromEnum(level) < @intFromEnum(self.level)) return;
        
        const timestamp = std.time.milliTimestamp() - self.start_time;
        const hours = @divTrunc(timestamp, 3600000);
        const minutes = @divTrunc(@mod(timestamp, 3600000), 60000);
        const seconds = @divTrunc(@mod(timestamp, 60000), 1000);
        const ms = @mod(timestamp, 1000);
        
        const time_str = try std.fmt.allocPrint(self.allocator, "{d:02}:{d:02}:{d:02}.{d:03}", .{
            hours, minutes, seconds, ms
        });
        defer self.allocator.free(time_str);
        
        const msg = try std.fmt.allocPrint(self.allocator, format, args);
        defer self.allocator.free(msg);
        
        const output = try std.fmt.allocPrint(self.allocator, "{s} [{s}] {s} - {s}\n", .{
            level.prefix(),
            time_str,
            level.britishPhrase(),
            msg,
        });
        defer self.allocator.free(output);
        
        // Write to stdout
        try std.io.getStdOut().writer().writeAll(output);
        
        // Write to file if present
        if (self.file) |f| {
            try f.writer().writeAll(output);
        }
    }
    
    pub fn debug(self: *Self, comptime format: []const u8, args: anytype) !void {
        try self.log(.debug, format, args);
    }
    
    pub fn info(self: *Self, comptime format: []const u8, args: anytype) !void {
        try self.log(.info, format, args);
    }
    
    pub fn warn(self: *Self, comptime format: []const u8, args: anytype) !void {
        try self.log(.warn, format, args);
    }
    
    pub fn err(self: *Self, comptime format: []const u8, args: anytype) !void {
        try self.log(.err, format, args);
    }
    
    pub fn fatal(self: *Self, comptime format: []const u8, args: anytype) !void {
        try self.log(.fatal, format, args);
        @panic("Oh dear, we must stop now.");
    }
};

test "logger - basic logging" {
    const allocator = std.testing.allocator;
    var logger = try Logger.init(allocator, .debug, null);
    defer logger.deinit();
    
    try logger.debug("Testing {d}", .{42});
    try logger.info("All {s}", .{"good"});
    try logger.warn("Warning {s}", .{"something"});
}
