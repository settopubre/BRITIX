// teatime.zig - Tea break scheduler for Britix
// "Three PM sharp. Earl Grey. No exceptions."

const std = @import("std");
const builtin = @import("builtin");
const time = std.time;

pub const TeaType = enum {
    earl_grey,
    english_breakfast,
    darjeeling,
    assam,
    oolong,
    green,
    white,
    herbal,
    
    pub fn description(self: TeaType) []const u8 {
        return switch (self) {
            .earl_grey => "Earl Grey with a hint of bergamot. Jolly good!",
            .english_breakfast => "Strong and robust. Perfect for coding.",
            .darjeeling => "The champagne of teas. Exquisite!",
            .assam => "Malty and bold. For serious thinking.",
            .oolong => "Partially oxidized. Deeply complex.",
            .green => "Delicate and fresh. Refreshing!",
            .white => "The least processed. Subtle and sweet.",
            .herbal => "No caffeine, but very soothing.",
        };
    }
};

pub const TeaSchedule = struct {
    const Self = @This();
    
    tea_time_hour: u32 = 15, // 3pm
    tea_time_minute: u32 = 0,
    default_tea: TeaType = .earl_grey,
    milk_first: bool = false, // Debate continues
    sugar_lumps: u32 = 0,
    
    pub fn init() TeaSchedule {
        return TeaSchedule{};
    }
    
    pub fn load_from_file(allocator: std.mem.Allocator, path: []const u8) !TeaSchedule {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        
        const content = try file.readToEndAlloc(allocator, 1024);
        defer allocator.free(content);
        
        var schedule = TeaSchedule{};
        
        // Parse simple format: "hour=15,minute=0,tea=earl_grey,milk_first=false,sugar=0"
        var lines = std.mem.splitSequence(u8, content, "\n");
        while (lines.next()) |line| {
            if (line.len == 0 or line[0] == '#') continue;
            
            var parts = std.mem.splitSequence(u8, line, "=");
            const key = parts.first();
            const value = parts.next() orelse continue;
            
            if (std.mem.eql(u8, key, "hour")) {
                schedule.tea_time_hour = std.fmt.parseInt(u32, value, 10) catch 15;
            } else if (std.mem.eql(u8, key, "minute")) {
                schedule.tea_time_minute = std.fmt.parseInt(u32, value, 10) catch 0;
            } else if (std.mem.eql(u8, key, "tea")) {
                schedule.default_tea = std.meta.stringToEnum(TeaType, value) orelse .earl_grey;
            } else if (std.mem.eql(u8, key, "milk_first")) {
                schedule.milk_first = std.mem.eql(u8, value, "true");
            } else if (std.mem.eql(u8, key, "sugar")) {
                schedule.sugar_lumps = std.fmt.parseInt(u32, value, 10) catch 0;
            }
        }
        
        return schedule;
    }
    
    pub fn is_tea_time(self: *const TeaSchedule) bool {
        const now = time.timestamp();
        const seconds_in_day = 24 * 60 * 60;
        const time_of_day = @mod(now, seconds_in_day);
        
        const tea_seconds = self.tea_time_hour * 3600 + self.tea_time_minute * 60;
        
        // Within 5 minutes of tea time
        return @abs(time_of_day - tea_seconds) < 300;
    }
    
    pub fn time_until_tea(self: *const TeaSchedule) i64 {
        const now = time.timestamp();
        const seconds_in_day = 24 * 60 * 60;
        const time_of_day = @mod(now, seconds_in_day);
        
        const tea_seconds = self.tea_time_hour * 3600 + self.tea_time_minute * 60;
        
        var diff = tea_seconds - time_of_day;
        if (diff < 0) diff += seconds_in_day;
        
        return diff;
    }
};

test "teatime - schedule parsing" {
    const allocator = std.testing.allocator;
    
    // Create temp schedule file
    const content = 
        \\hour=16
        \\minute=30
        \\tea=darjeeling
        \\milk_first=false
        \\sugar=1
    ;
    
    const tmp_path = "/tmp/tea_test.txt";
    const file = try std.fs.cwd().createFile(tmp_path, .{});
    defer file.close();
    defer std.fs.cwd().deleteFile(tmp_path) catch {};
    
    try file.writeAll(content);
    
    const schedule = try TeaSchedule.load_from_file(allocator, tmp_path);
    try std.testing.expectEqual(@as(u32, 16), schedule.tea_time_hour);
    try std.testing.expectEqual(@as(u32, 30), schedule.tea_time_minute);
    try std.testing.expectEqual(TeaType.darjeeling, schedule.default_tea);
    try std.testing.expectEqual(@as(u32, 1), schedule.sugar_lumps);
}
