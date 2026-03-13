// random.zig - Xoroshiro128+ deterministic random number generator
// "Random, but make it reproducible! Jolly good for sampling!"

const std = @import("std");

pub const Xoroshiro128Plus = struct {
    const Self = @This();
    
    s0: u64,
    s1: u64,
    
    pub fn init(seed: u64) Self {
        // SplitMix64 to initialize state
        var splitmix = SplitMix64.init(seed);
        return Self{
            .s0 = splitmix.next(),
            .s1 = splitmix.next(),
        };
    }
    
    pub fn next(self: *Self) u64 {
        const s0 = self.s0;
        var s1 = self.s1;
        const result = s0 +% s1;
        
        s1 ^= s0;
        self.s0 = std.math.rotl(u64, s0, 24) ^ s1 ^ (s1 << 16);
        self.s1 = std.math.rotl(u64, s1, 37);
        
        return result;
    }
    
    pub fn nextF32(self: *Self) f32 {
        return @as(f32, @floatFromInt(self.next() >> 40)) / @as(f32, @floatFromInt(1 << 24));
    }
    
    pub fn nextF64(self: *Self) f64 {
        return @as(f64, @floatFromInt(self.next() >> 11)) / @as(f64, @floatFromInt(1 << 53));
    }
    
    pub fn nextRange(self: *Self, min: u64, max: u64) u64 {
        std.debug.assert(max > min);
        const range = max - min;
        return min + (self.next() % range);
    }
    
    pub fn shuffle(self: *Self, comptime T: type, slice: []T) void {
        var i = slice.len;
        while (i > 1) {
            i -= 1;
            const j = self.nextRange(0, @as(u64, @intCast(i + 1)));
            std.mem.swap(T, &slice[i], &slice[j]);
        }
    }
};

const SplitMix64 = struct {
    state: u64,
    
    fn init(seed: u64) SplitMix64 {
        return SplitMix64{ .state = seed };
    }
    
    fn next(self: *SplitMix64) u64 {
        self.state +%= 0x9e3779b97f4a7c15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }
};

pub const Sampler = struct {
    rng: Xoroshiro128Plus,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    
    pub fn init(seed: u64, temperature: f32, top_k: usize, top_p: f32) Sampler {
        return Sampler{
            .rng = Xoroshiro128Plus.init(seed),
            .temperature = temperature,
            .top_k = top_k,
            .top_p = top_p,
        };
    }
    
    pub fn sample(self: *Sampler, allocator: std.mem.Allocator, logits: []f32) !usize {
        var probs = logits;
        
        // Apply temperature
        if (self.temperature > 0 and self.temperature != 1.0) {
            const inv_temp = 1.0 / self.temperature;
            for (probs) |*p| {
                p.* *= inv_temp;
            }
        }
        
        // Apply softmax
        @import("math.zig").softmax(probs);
        
        // Top-k filtering
        if (self.top_k > 0 and self.top_k < probs.len) {
            var indices = try allocator.alloc(usize, probs.len);
            defer allocator.free(indices);
            
            for (0..probs.len) |i| indices[i] = i;
            std.sort.insertion(usize, indices, probs, struct {
                fn lessThan(ctx: []f32, a: usize, b: usize) bool {
                    return ctx[a] > ctx[b];
                }
            }.lessThan);
            
            // Zero out probabilities below top-k
            for (indices[self.top_k..]) |idx| {
                probs[idx] = 0;
            }
            
            // Renormalize
            var sum: f32 = 0;
            for (probs) |p| sum += p;
            const inv_sum = 1.0 / sum;
            for (probs) |*p| p.* *= inv_sum;
        }
        
        // Sample
        const r = self.rng.nextF32();
        var cumsum: f32 = 0;
        for (probs, 0..) |p, i| {
            cumsum += p;
            if (r < cumsum) {
                return i;
            }
        }
        
        return probs.len - 1;
    }
    
    pub fn sampleGreedy(_: *Sampler, logits: []f32) usize {
        var max_idx: usize = 0;
        var max_val: f32 = logits[0];
        
        for (logits[1..], 1..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        
        return max_idx;
    }
};

test "random - determinism" {
    var rng1 = Xoroshiro128Plus.init(1234);
    var rng2 = Xoroshiro128Plus.init(1234);
    
    for (0..100) |_| {
        try std.testing.expectEqual(rng1.next(), rng2.next());
    }
}

test "random - distribution" {
    var rng = Xoroshiro128Plus.init(5678);
    
    var counts = [_]usize{0} ** 10;
    const iterations = 10000;
    
    for (0..iterations) |_| {
        const val = rng.nextRange(0, 10);
        counts[val] += 1;
    }
    
    // Each bucket should have roughly 1000
    for (counts) |c| {
        try std.testing.expect(c > 800 and c < 1200);
    }
}
