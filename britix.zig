// britix.zig - BRITIX 8B Core System
const std = @import("std");
const tts = @import("005_TTS_ENGINE.zig");
const inference = @import("inference.zig");
const parameters = @import("parameters.zig");
const tokenizer = @import("tokenizer.zig");
const sqlite = @import("sqlite.zig");

pub const BritixSystem = struct {
    allocator: std.mem.Allocator,
    tts_engine: ?*tts.TTSEngine,
    inference_state: ?*inference.InferenceState,
    params: ?*parameters.Parameters,
    tokenizer: ?*tokenizer.Tokenizer,
    memory: ?ConversationMemory,
    weights_data: []align(std.mem.page_size) u8,
    weights_file: std.fs.File,
    response_cache: std.AutoHashMap(u64, []u8),
    running: bool,
    conversation_id: ?i64,
    
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        const file = try std.fs.cwd().openFile("weights.bin", .{});
        const size = try file.getEndPos();
        const mapped = try std.posix.mmap(
            null,
            size,
            std.posix.PROT.READ,
            std.posix.MAP{ .TYPE = .PRIVATE },
            file.handle,
            0
        );
        
        var self = Self{
            .allocator = allocator,
            .tts_engine = null,
            .inference_state = null,
            .params = null,
            .tokenizer = null,
            .memory = null,
            .weights_data = mapped,
            .weights_file = file,
            .response_cache = std.AutoHashMap(u64, []u8).init(allocator),
            .running = true,
            .conversation_id = null,
        };
        errdefer self.deinit();

        var wrapped = parameters.MappedWeights{
            .data = self.weights_data,
            .file = self.weights_file,
            .allocator = allocator,
        };

        const params_ptr = try allocator.create(parameters.Parameters);
        errdefer allocator.destroy(params_ptr);
        params_ptr.* = try parameters.loadFromMappedFile(allocator, &wrapped);
        self.params = params_ptr;

        const tok_ptr = try allocator.create(tokenizer.Tokenizer);
        tok_ptr.* = tokenizer.Tokenizer.init(allocator);
        
        tok_ptr.load_vocab("vocab.json") catch |err| {
            tok_ptr.deinit();
            allocator.destroy(tok_ptr);
            self.params = null;
            std.debug.print("❌ Failed to load vocab.json: {}\n", .{err});
            return err;
        };
        
        tok_ptr.load_merges("merges.txt") catch |err| {
            tok_ptr.deinit();
            allocator.destroy(tok_ptr);
            self.params = null;
            std.debug.print("❌ Failed to load merges.txt: {}\n", .{err});
            return err;
        };
        
        self.tokenizer = tok_ptr;

        const inf_ptr = try allocator.create(inference.InferenceState);
        errdefer allocator.destroy(inf_ptr);
        inf_ptr.* = try inference.InferenceState.init(
            allocator, 
            params_ptr, 
            tok_ptr, 
            .{
                .temperature = 0.7,
                .top_k = 40,
                .top_p = 0.9,
                .max_tokens = 2048,
            }
        );
        self.inference_state = inf_ptr;

        const tts_ptr = try allocator.create(tts.TTSEngine);
        errdefer allocator.destroy(tts_ptr);
        tts_ptr.* = try tts.TTSEngine.init(allocator);
        self.tts_engine = tts_ptr;

        const memory = try ConversationMemory.init(allocator, "britix_memory.db");
        self.memory = memory;

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.memory) |*mem| mem.deinit();
        if (self.tts_engine) |ptr| {
            ptr.deinit();
            self.allocator.destroy(ptr);
        }
        if (self.inference_state) |ptr| {
            ptr.deinit();
            self.allocator.destroy(ptr);
        }
        if (self.tokenizer) |ptr| {
            ptr.deinit();
            self.allocator.destroy(ptr);
        }
        if (self.params) |ptr| {
            ptr.deinit();
            self.allocator.destroy(ptr);
        }

        var it = self.response_cache.iterator();
        while (it.next()) |entry| self.allocator.free(entry.value_ptr.*);
        self.response_cache.deinit();

        std.posix.munmap(self.weights_data);
        self.weights_file.close();
    }

    pub fn generate(self: *Self, prompt: []const u8) ![]u8 {
        std.debug.print("🤔 Thinking...\n", .{});
        
        const inf_state = self.inference_state orelse return error.InferenceNotInitialized;
        const tok = self.tokenizer orelse return error.TokenizerNotInitialized;
        const tts_eng = self.tts_engine orelse return error.TTSNotInitialized;
        const params = self.params orelse return error.ParametersNotInitialized;
        const mem = &(self.memory orelse return error.MemoryNotInitialized);
        
        _ = tts_eng;
        _ = params;
        _ = tok;
        
        const hash = std.hash.Wyhash.hash(0, prompt);
        if (self.response_cache.get(hash)) |cached| {
            std.debug.print("📝 [CACHED] {s}\n", .{cached});
            return try self.allocator.dupe(u8, cached);
        }
        
        if (self.conversation_id == null) self.conversation_id = try mem.startConversation();

        const history = try mem.getConversationHistory(self.conversation_id.?, 5);
        defer {
            for (history) |h| self.allocator.free(h);
            self.allocator.free(history);
        }

        var context = try std.ArrayList(u8).initCapacity(self.allocator, 1024);
        defer context.deinit();
        const writer = context.writer();

        try writer.print(
            \\[SYSTEM: You are Britix, a helpful AI assistant with a British accent.]
            \\[SYSTEM: Address user as "Governor".]
            \\[SYSTEM: Respond in 2-3 natural sentences.]
            \\[SYSTEM: Tea is at 3pm, Earl Grey, no sugar.]
            \\
        , .{});

        if (history.len > 0) {
            try writer.print("\n[Previous]:\n", .{});
            for (history) |line| try writer.print("{s}\n", .{line});
        }

        try writer.print("\nGovernor: {s}\nBritix: ", .{prompt});

        const ai_response = try inf_state.generate(context.items);
        defer self.allocator.free(ai_response);

        const final_response = try self.cleanResponse(ai_response, prompt) orelse 
            return try self.allocator.dupe(u8, "I'm here to assist you, Governor.");

        // Print the actual response
        std.debug.print("\n", .{});
        std.debug.print("{s}\n", .{final_response});

        const cached_response = try self.allocator.dupe(u8, final_response);
        errdefer self.allocator.free(cached_response);
        try self.response_cache.put(hash, cached_response);

        try mem.addToConversation(self.conversation_id.?, prompt, final_response);
        return try self.allocator.dupe(u8, final_response);
    }

    fn cleanResponse(self: *Self, response: []const u8, user_request: []const u8) !?[]u8 {
        var cleaned = response;
        if (std.mem.indexOf(u8, cleaned, "Britix:")) |idx| cleaned = cleaned[idx+7..];
        
        const bad = [_][]const u8{
            "2-3 sentences", "as requested", "Direct reply",
            "I will do my best", "Your request is noted",
        };
        
        for (bad) |phrase| {
            if (std.mem.indexOf(u8, cleaned, phrase) != null) {
                var lower_buf: [256]u8 = undefined;
                const lower = std.ascii.lowerString(&lower_buf, user_request);
                
                if (std.mem.indexOf(u8, lower, "tea") != null)
                    return try self.allocator.dupe(u8, "Tea time is at 3pm sharp, Governor. Earl Grey, no sugar.");
                if (std.mem.indexOf(u8, lower, "hello") != null)
                    return try self.allocator.dupe(u8, "Good day, Governor. How may I assist?");
                return try self.allocator.dupe(u8, "I'm here to help, Governor. What do you need?");
            }
        }
        
        cleaned = std.mem.trim(u8, cleaned, " \t\n\r.,:;");
        if (cleaned.len < 5) return null;
        return try self.allocator.dupe(u8, cleaned);
    }

    pub fn processCommand(self: *Self, input: []const u8) !void {
        const trimmed = std.mem.trim(u8, input, " \t\r\n");
        if (trimmed.len == 0) return;

        const tts_eng = self.tts_engine orelse return error.TTSNotInitialized;
        const mem = &(self.memory orelse return error.MemoryNotInitialized);

        if (std.mem.eql(u8, trimmed, "!recall")) return self.recallRecent();
        if (std.mem.eql(u8, trimmed, "!forget")) {
            try mem.clearAllMemories();
            try tts_eng.speak("Memory banks wiped clean, Governor.");
            return;
        }
        if (std.mem.eql(u8, trimmed, "!tea")) {
            try tts_eng.speak("Earl Grey, no sugar. Jolly good timing, Governor!");
            return;
        }
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) {
            try tts_eng.speak("Cheerio, Governor! Come back for tea anytime.");
            self.running = false;
            return;
        }

        const response = try self.generate(trimmed);
        defer self.allocator.free(response);
        
        // Also speak it if TTS is enabled
        try tts_eng.speak(response);
    }

    fn recallRecent(self: *Self) !void {
        const tts_eng = self.tts_engine orelse return error.TTSNotInitialized;
        const mem = &(self.memory orelse return error.MemoryNotInitialized);
        
        if (self.conversation_id == null) {
            try tts_eng.speak("No active conversation, Governor.");
            return;
        }

        const history = try mem.getConversationHistory(self.conversation_id.?, 10);
        defer {
            for (history) |h| self.allocator.free(h);
            self.allocator.free(history);
        }

        if (history.len == 0) {
            try tts_eng.speak("No recent memories, Governor.");
            return;
        }

        var response = try std.ArrayList(u8).initCapacity(self.allocator, 512);
        defer response.deinit();

        try response.writer().print("Recent memories:\n", .{});
        for (history) |line| try response.writer().print("{s}\n", .{line});

        try tts_eng.speak(response.items);
    }
};

const ConversationMemory = struct {
    allocator: std.mem.Allocator,
    db: sqlite.Db,
    
    pub fn init(allocator: std.mem.Allocator, db_path: []const u8) !ConversationMemory {
        var db = try sqlite.Db.init(.{
            .mode = sqlite.Db.Mode.File,
            .flags = .{ .create = true, .read_write = true },
        });
        errdefer db.deinit();

        try db.open(db_path);
        try db.exec("PRAGMA journal_mode=WAL;", .{});

        try db.exec(
            \\CREATE TABLE IF NOT EXISTS conversations (
            \\    id INTEGER PRIMARY KEY AUTOINCREMENT,
            \\    start_time INTEGER NOT NULL
            \\);
            \\
            \\CREATE TABLE IF NOT EXISTS messages (
            \\    id INTEGER PRIMARY KEY AUTOINCREMENT,
            \\    conversation_id INTEGER NOT NULL,
            \\    timestamp INTEGER NOT NULL,
            \\    role TEXT NOT NULL,
            \\    content TEXT NOT NULL
            \\);
            \\
            \\CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
        , .{});

        return ConversationMemory{ .allocator = allocator, .db = db };
    }

    pub fn deinit(self: *ConversationMemory) void { self.db.deinit(); }
    
    pub fn startConversation(self: *ConversationMemory) !i64 {
        const timestamp = std.time.timestamp();
        var stmt = try self.db.prepare("INSERT INTO conversations (start_time) VALUES (?);");
        defer stmt.deinit();
        try stmt.bind(.{timestamp});
        _ = try stmt.step();
        
        var last_stmt = try self.db.prepare("SELECT last_insert_rowid();");
        defer last_stmt.deinit();
        _ = try last_stmt.step();
        return last_stmt.int(0);
    }

    pub fn addToConversation(self: *ConversationMemory, conv_id: i64, user_msg: []const u8, assistant_msg: []const u8) !void {
        const timestamp = std.time.timestamp();
        var user_stmt = try self.db.prepare("INSERT INTO messages (conversation_id, timestamp, role, content) VALUES (?, ?, 'user', ?);");
        defer user_stmt.deinit();
        try user_stmt.bind(.{ conv_id, timestamp, user_msg });
        _ = try user_stmt.step();

        var asst_stmt = try self.db.prepare("INSERT INTO messages (conversation_id, timestamp, role, content) VALUES (?, ?, 'assistant', ?);");
        defer asst_stmt.deinit();
        try asst_stmt.bind(.{ conv_id, timestamp, assistant_msg });
        _ = try asst_stmt.step();
    }

    pub fn getConversationHistory(self: *ConversationMemory, conv_id: i64, limit: usize) ![][]const u8 {
        var stmt = try self.db.prepare(
            \\SELECT role, content FROM messages 
            \\WHERE conversation_id = ? ORDER BY timestamp ASC LIMIT ?;
        );
        defer stmt.deinit();
        try stmt.bind(.{ conv_id, @as(i64, @intCast(limit)) });

        var results = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (results.items) |r| self.allocator.free(r);
            results.deinit();
        }

        while (try stmt.step()) {
            const role = try stmt.text(0);
            const content = try stmt.text(1);
            const formatted = try std.fmt.allocPrint(self.allocator, "{s}: {s}", .{ role, content });
            try results.append(formatted);
        }
        return results.toOwnedSlice();
    }

    pub fn clearAllMemories(self: *ConversationMemory) !void {
        try self.db.exec("DELETE FROM messages;", .{});
        try self.db.exec("DELETE FROM conversations;", .{});
        try self.db.exec("VACUUM;", .{});
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("╔════════════════════════════════════════╗\n", .{});
    std.debug.print("║          BRITIX 8B v1.0                ║\n", .{});
    std.debug.print("╚════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});

    var britix = BritixSystem.init(allocator) catch |err| {
        std.debug.print("❌ Failed to initialize Britix: {}\n", .{err});
        return;
    };
    defer britix.deinit();
    
    std.debug.print("✅ Britix ready! Type 'exit' to quit.\n\n", .{});

    const stdin = std.io.getStdIn().reader();
    var buffer: [1024]u8 = undefined;

    while (britix.running) {
        std.debug.print("> ", .{});
        const line = stdin.readUntilDelimiterOrEof(&buffer, '\n') catch continue;
        if (line) |input| {
            britix.processCommand(input) catch |err| {
                std.debug.print("❌ Error: {}\n", .{err});
            };
        }
    }
    
    std.debug.print("\n👋 Goodbye, Governor!\n", .{});
}
