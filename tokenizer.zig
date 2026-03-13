// tokenizer.zig - BPE tokenizer for Britix (Qwen2.5 compatible)
// "Turning words into numbers, and back again. Jolly good!"

const std = @import("std");
const builtin = @import("builtin");
const json = std.json;

pub const Tokenizer = struct {
    const Self = @This();
    
    vocab: std.StringHashMap(u32),
    reverse_vocab: std.AutoHashMap(u32, []u8),
    merges: std.ArrayList(BPEMerge),
    cache: std.StringHashMap([]u32),
    allocator: std.mem.Allocator,
    
    // Qwen2.5 uses ▁ (U+2581) as word-start prefix
    pub const WORD_START_PREFIX = "\u{2581}";
    
    pub const BPEMerge = struct {
        left: []const u8,
        right: []const u8,
        priority: u32,
    };
    
    pub fn init(allocator: std.mem.Allocator) Tokenizer {
        return Tokenizer{
            .vocab = std.StringHashMap(u32).init(allocator),
            .reverse_vocab = std.AutoHashMap(u32, []u8).init(allocator),
            .merges = std.ArrayList(BPEMerge).init(allocator),
            .cache = std.StringHashMap([]u32).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        // Track pointers we've already freed to prevent double-free
        var freed = std.AutoHashMap(usize, void).init(self.allocator);
        defer freed.deinit();
        
        // Free vocab keys
        var vocab_iter = self.vocab.keyIterator();
        while (vocab_iter.next()) |key| {
            const ptr = @intFromPtr(key.*.ptr);
            freed.put(ptr, {}) catch {};
            self.allocator.free(key.*);
        }
        
        // Free reverse_vocab values - but only if not already freed
        var rev_iter = self.reverse_vocab.valueIterator();
        while (rev_iter.next()) |val| {
            const ptr = @intFromPtr(val.*.ptr);
            if (!freed.contains(ptr)) {
                self.allocator.free(val.*);
            }
        }
        
        // Free merges
        for (self.merges.items) |merge| {
            self.allocator.free(merge.left);
            self.allocator.free(merge.right);
        }
        
        // Clear cache values
        var cache_iter = self.cache.valueIterator();
        while (cache_iter.next()) |val| {
            self.allocator.free(val.*);
        }
        
        // Free cache keys
        var cache_key_iter = self.cache.keyIterator();
        while (cache_key_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        
        self.vocab.deinit();
        self.reverse_vocab.deinit();
        self.merges.deinit();
        self.cache.deinit();
    }
    
    pub fn load_vocab(self: *Self, vocab_path: []const u8) !void {
        const file = try std.fs.cwd().openFile(vocab_path, .{});
        defer file.close();
        const content = try file.readToEndAlloc(self.allocator, 50 * 1024 * 1024);
        defer self.allocator.free(content);
        
        var parsed = try json.parseFromSlice(json.Value, self.allocator, content, .{});
        defer parsed.deinit();
        
        const root = parsed.value;
        const dict = root.object;
        var iter = dict.iterator();
        while (iter.next()) |entry| {
            const token = try self.allocator.dupe(u8, entry.key_ptr.*);
            const id = @as(u32, @intCast(entry.value_ptr.*.integer));
            try self.vocab.put(token, id);
            try self.reverse_vocab.put(id, token);
        }
    }
    
    pub fn load_merges(self: *Self, merges_path: []const u8) !void {
        const file = try std.fs.cwd().openFile(merges_path, .{});
        defer file.close();
        var reader = file.reader();
        var line_buffer: [2048]u8 = undefined;
        _ = try reader.readUntilDelimiterOrEof(&line_buffer, '\n'); // skip version
        
        var priority: u32 = 0;
        while (try reader.readUntilDelimiterOrEof(&line_buffer, '\n')) |line| {
            if (line.len == 0) continue;
            var parts = std.mem.splitSequence(u8, line, " ");
            const left = parts.first();
            const right = parts.next() orelse continue;
            try self.merges.append(BPEMerge{
                .left = try self.allocator.dupe(u8, left),
                .right = try self.allocator.dupe(u8, right),
                .priority = priority,
            });
            priority += 1;
        }
    }
    
    // Helper: lookup token with/without prefix
    fn lookupToken(self: *Self, word: []const u8, is_word_start: bool) ?u32 {
        // Try with prefix first (most common case)
        if (is_word_start) {
            var prefixed: [512]u8 = undefined;
            if (WORD_START_PREFIX.len + word.len <= prefixed.len) {
                @memcpy(prefixed[0..WORD_START_PREFIX.len], WORD_START_PREFIX);
                @memcpy(prefixed[WORD_START_PREFIX.len..][0..word.len], word);
                if (self.vocab.get(prefixed[0..WORD_START_PREFIX.len + word.len])) |id| {
                    return id;
                }
            }
        }
        // Fallback: try without prefix
        return self.vocab.get(word);
    }
    
    pub fn encode(self: *Self, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        if (text.len == 0) return try allocator.alloc(u32, 0);
        
        // Check cache
        if (self.cache.get(text)) |cached| {
            return try allocator.dupe(u32, cached);
        }
        
        var tokens = std.ArrayList(u32).init(allocator);
        errdefer tokens.deinit();
        
        // Process the entire text character by character with BPE
        var i: usize = 0;
        var is_word_start = true;
        
        while (i < text.len) {
            // Try to find the longest matching token starting at i
            var longest_match: ?u32 = null;
            var match_len: usize = 1;
            
            // Try matches up to 50 chars (max token length)
            var j = @min(i + 50, text.len);
            while (j > i) : (j -= 1) {
                const candidate = text[i..j];
                
                // Try with prefix if at word start
                if (is_word_start) {
                    var prefixed: [512]u8 = undefined;
                    if (WORD_START_PREFIX.len + candidate.len <= prefixed.len) {
                        @memcpy(prefixed[0..WORD_START_PREFIX.len], WORD_START_PREFIX);
                        @memcpy(prefixed[WORD_START_PREFIX.len..][0..candidate.len], candidate);
                        const prefixed_candidate = prefixed[0..WORD_START_PREFIX.len + candidate.len];
                        
                        if (self.vocab.get(prefixed_candidate)) |id| {
                            longest_match = id;
                            match_len = j - i;
                            break;
                        }
                    }
                }
                
                // Try without prefix
                if (self.vocab.get(candidate)) |id| {
                    longest_match = id;
                    match_len = j - i;
                    break;
                }
            }
            
            if (longest_match) |id| {
                try tokens.append(id);
                i += match_len;
                is_word_start = false;
            } else {
                // No match found - use unknown token
                try tokens.append(0); // <unk>
                i += 1;
                is_word_start = false;
            }
            
            // Check if next character is whitespace to reset word start
            if (i < text.len and std.ascii.isWhitespace(text[i])) {
                is_word_start = true;
            }
        }
        
        const result = try tokens.toOwnedSlice();
        
        // Cache result
        const text_copy = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(text_copy);
        const result_copy = try self.allocator.dupe(u32, result);
        errdefer self.allocator.free(result_copy);
        try self.cache.put(text_copy, result_copy);
        
        return result;
    }
    
    pub fn decode(self: *Self, allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
        var result = std.ArrayList(u8).init(allocator);
        defer result.deinit();
        
        for (tokens) |token| {
            if (self.reverse_vocab.get(token)) |word| {
                // Replace ▁ with space for output
                if (std.mem.startsWith(u8, word, WORD_START_PREFIX)) {
                    if (result.items.len == 0) {
                        try result.appendSlice(word[WORD_START_PREFIX.len..]);
                    } else {
                        try result.append(' ');
                        try result.appendSlice(word[WORD_START_PREFIX.len..]);
                    }
                } else {
                    try result.appendSlice(word);
                }
            } else {
                try result.appendSlice("<unk>");
            }
        }
        return try result.toOwnedSlice();
    }
    
    pub fn vocab_size(self: *Self) u32 {
        return @as(u32, @intCast(self.vocab.count()));
    }
    
    // Debug helper: print token IDs for a string
    pub fn debugEncode(self: *Self, text: []const u8) void {
        std.debug.print("🔍 encode('{s}') → ", .{text});
        const tokens = self.encode(self.allocator, text) catch |err| {
            std.debug.print("ERROR: {any}\n", .{err});
            return;
        };
        defer self.allocator.free(tokens);
        for (tokens) |t| std.debug.print("{d} ", .{t});
        std.debug.print("\n", .{});
    }
};

test "tokenizer - init and deinit" {
    const allocator = std.testing.allocator;
    var tokenizer = Tokenizer.init(allocator);
    defer tokenizer.deinit();
    try std.testing.expectEqual(@as(usize, 0), tokenizer.vocab.count());
}

test "tokenizer - prefix lookup" {
    const allocator = std.testing.allocator;
    var tokenizer = Tokenizer.init(allocator);
    defer tokenizer.deinit();
    
    // Add token with prefix: ▁hello = 21558
    const prefixed = try allocator.dupe(u8, "\u{2581}hello");
    try tokenizer.vocab.put(prefixed, 21558);
    try tokenizer.reverse_vocab.put(21558, prefixed);
    
    // Lookup with word_start=true should find it
    const id = tokenizer.lookupToken("hello", true);
    try std.testing.expectEqual(@as(?u32, 21558), id);
}
