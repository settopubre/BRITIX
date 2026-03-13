// 005_TTS_ENGINE.zig - Text-to-Speech for Britix
// "The voice of British AI. Jolly good, what what!"

const std = @import("std");

pub const TTSEngine = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) !TTSEngine {
        return TTSEngine{
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *TTSEngine) void {
        _ = self;
    }
    
    pub fn speak(self: *TTSEngine, text: []const u8) !void {
        _ = self;
        std.debug.print("🔊 Britix says: {s}\n", .{text});
        
        // MBROLA EN1 - The voice of British gentlemen
        var child = std.process.Child.init(&[_][]const u8{
            "espeak-ng", "-v", "mb-en1", "-s", "150", "-p", "50", text
        }, std.heap.page_allocator);
        
        child.stdout_behavior = .Ignore;
        child.stderr_behavior = .Ignore;
        
        child.spawn() catch {
            std.debug.print("   (espeak-ng not installed - text only)\n", .{});
            return;
        };
        _ = child.wait() catch {};
    }
    
    pub fn getGreeting(self: *TTSEngine) ![]const u8 {
        const greetings = [_][]const u8{
            "Good day, Governor.",
            "Greetings, Governor. Britix at your service.",
            "Ah, Governor. A pleasure to see you.",
            "Right then, Governor. Ready to assist.",
            "Jolly good to see you, Governor.",
            "Tea time soon, Governor. Earl Grey, I presume?",
            "What what! Ready when you are, Governor.",
        };
        const idx = @mod(@as(usize, @intCast(std.time.timestamp())), greetings.len);
        return try self.allocator.dupe(u8, greetings[idx]);
    }
    
    pub fn getSignature(self: *TTSEngine) ![]const u8 {
        return try self.allocator.dupe(u8, "This is Britix, your formally verified assistant. I have spoken.");
    }
    
    pub fn announceTeaTime(self: *TTSEngine) ![]const u8 {
        const tea_msgs = [_][]const u8{
            "Three PM, Governor. Tea time! Earl Grey, no sugar.",
            "Ding! Tea is served, Governor. Jolly good!",
            "The kettle's boiled, Governor. Shall I pour?",
            "Tea time, what what! The proper British break.",
        };
        const idx = @mod(@as(usize, @intCast(std.time.timestamp())), tea_msgs.len);
        return try self.allocator.dupe(u8, tea_msgs[idx]);
    }
};
