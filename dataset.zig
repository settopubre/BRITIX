// dataset.zig - Data loading for Britix training
// "Organized knowledge, like a proper British library."

const std = @import("std");
const builtin = @import("builtin");
const mem = std.mem;

pub const Batch = struct {
    inputs: []u32,
    targets: []u32,
    
    pub fn deinit(self: *Batch, allocator: std.mem.Allocator) void {
        allocator.free(self.inputs);
        allocator.free(self.targets);
    }
};

pub const Dataset = struct {
    const Self = @This();
    
    // Data loaded into memory
    token_ids: []u32,
    
    // Metadata
    sequence_length: usize,
    num_tokens: usize,
    num_sequences: usize,
    
    // Current position
    position: usize = 0,
    
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, path: []const u8, sequence_length: usize) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        
        const file_size = try file.getEndPos();
        const file_contents = try file.readToEndAlloc(allocator, file_size);
        defer allocator.free(file_contents);
        
        // Interpret as u32 tokens (assuming file contains u32 token IDs)
        const token_ids = try allocator.alloc(u32, file_size / @sizeOf(u32));
        errdefer allocator.free(token_ids);
        
        @memcpy(std.mem.sliceAsBytes(token_ids), file_contents);
        
        return Self{
            .token_ids = token_ids,
            .sequence_length = sequence_length,
            .num_tokens = token_ids.len,
            .num_sequences = if (token_ids.len > sequence_length) token_ids.len - sequence_length else 0,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.token_ids);
    }
    
    pub fn get_batch(self: *Self, allocator: std.mem.Allocator, batch_size: usize) !Batch {
        if (self.num_sequences == 0) return error.NoSequences;
        
        const total_tokens = batch_size * self.sequence_length;
        
        var inputs = try allocator.alloc(u32, total_tokens);
        errdefer allocator.free(inputs);
        
        var targets = try allocator.alloc(u32, total_tokens);
        errdefer allocator.free(targets);
        
        var pos = self.position;
        
        for (0..batch_size) |i| {
            const start = pos % self.num_sequences;
            const input_start = start;
            const target_start = start + 1;
            
            // Copy input sequence
            @memcpy(
                inputs[i * self.sequence_length ..][0..self.sequence_length],
                self.token_ids[input_start .. input_start + self.sequence_length]
            );
            
            // Copy target sequence (shifted by 1)
            @memcpy(
                targets[i * self.sequence_length ..][0..self.sequence_length],
                self.token_ids[target_start .. target_start + self.sequence_length]
            );
            
            pos += self.sequence_length;
        }
        
        self.position = pos;
        
        return Batch{
            .inputs = inputs,
            .targets = targets,
        };
    }
    
    pub fn reset(self: *Self) void {
        self.position = 0;
    }
};

pub const DataLoader = struct {
    const Self = @This();
    
    dataset: *Dataset,
    batch_size: usize,
    shuffle: bool,
    
    pub fn init(dataset: *Dataset, batch_size: usize, shuffle: bool) Self {
        return Self{
            .dataset = dataset,
            .batch_size = batch_size,
            .shuffle = shuffle,
        };
    }
    
    pub fn next(self: *Self, allocator: std.mem.Allocator) !?Batch {
        if (self.dataset.position >= self.dataset.num_sequences) {
            if (self.shuffle) {
                self.dataset.reset();
                // Optionally shuffle
            } else {
                return null;
            }
        }
        
        return try self.dataset.get_batch(allocator, self.batch_size);
    }
};

pub fn create_synthetic_dataset(allocator: std.mem.Allocator, num_tokens: usize, vocab_size: u32) ![]u32 {
    var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    
    var data = try allocator.alloc(u32, num_tokens);
    for (0..num_tokens) |i| {
        data[i] = rng.random().uintLessThan(u32, vocab_size);
    }
    
    return data;
}

test "dataset - synthetic data" {
    const allocator = std.testing.allocator;
    
    // Create synthetic data
    const data = try create_synthetic_dataset(allocator, 1000, 32000);
    defer allocator.free(data);
    
    try std.testing.expectEqual(@as(usize, 1000), data.len);
}

test "dataset - batch creation" {
    const allocator = std.testing.allocator;
    
    // Create simple sequential data for testing
    var data = try allocator.alloc(u32, 100);
    defer allocator.free(data);
    
    for (0..100) |i| {
        data[i] = @as(u32, @intCast(i));
    }
    
    // Write to temp file
    const tmp_path = "/tmp/britix_test.bin";
    const file = try std.fs.cwd().createFile(tmp_path, .{});
    defer file.close();
    defer std.fs.cwd().deleteFile(tmp_path) catch {};
    
    try file.writeAll(std.mem.sliceAsBytes(data));
    
    // Test dataset
    var dataset = try Dataset.init(allocator, tmp_path, 10);
    defer dataset.deinit();
    
    var batch = try dataset.get_batch(allocator, 2);
    defer batch.deinit(allocator);
    
    try std.testing.expectEqual(@as(usize, 20), batch.inputs.len);
    try std.testing.expectEqual(@as(usize, 20), batch.targets.len);
}
