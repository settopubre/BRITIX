// chat.zig - Interactive chat interface for Britix
// "Tea and conversation. What could be better?"

const std = @import("std");
const builtin = @import("builtin");

pub const Role = enum {
    user,
    assistant,
    system,
};

pub const ChatMessage = struct {
    role: Role,
    content: []u8,
    
    pub fn deinit(self: *ChatMessage, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
    }
};

pub const ChatHistory = struct {
    const Self = @This();
    
    messages: std.ArrayList(ChatMessage),
    max_history: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, max_history: usize) Self {
        return Self{
            .messages = std.ArrayList(ChatMessage).init(allocator),
            .max_history = max_history,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        for (self.messages.items) |*msg| {
            msg.deinit(self.allocator);
        }
        self.messages.deinit();
    }
    
    pub fn add(self: *Self, role: Role, content: []const u8) !void {
        // Remove oldest if at limit
        if (self.messages.items.len >= self.max_history) {
            var oldest = self.messages.orderedRemove(0);
            oldest.deinit(self.allocator);
        }
        
        const msg = ChatMessage{
            .role = role,
            .content = try self.allocator.dupe(u8, content),
        };
        try self.messages.append(msg);
    }
};

test "chat - history management" {
    const allocator = std.testing.allocator;
    var history = ChatHistory.init(allocator, 2);
    defer history.deinit();
    
    try history.add(.user, "Hello");
    try history.add(.assistant, "Hi there!");
    try history.add(.user, "How are you?");
    
    try std.testing.expectEqual(@as(usize, 2), history.messages.items.len);
    try std.testing.expectEqualStrings("Hi there!", history.messages.items[0].content);
    try std.testing.expectEqualStrings("How are you?", history.messages.items[1].content);
}
