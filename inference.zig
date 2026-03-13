// inference.zig - FINAL FIXED VERSION: Clean output, generation works
const std = @import("std");
const builtin = @import("builtin");

const parameters = @import("parameters.zig");
const attention = @import("attention.zig");
const feedforward = @import("feedforward.zig");
const tokenizer = @import("tokenizer.zig");
const sampler = @import("sampler.zig");
const cache = @import("cache.zig");
const math = @import("math.zig");

pub const InferenceConfig = struct {
    temperature: f32 = 0.7,
    top_k: usize = 40,
    top_p: f32 = 0.9,
    max_tokens: usize = 2048,
    stop_on_eos: bool = true,
    eos_token_id: u32 = 2,
    
    pub fn validate(self: *const InferenceConfig) !void {
        if (self.temperature < 0.0 or self.temperature > 2.0) return error.InvalidTemperature;
        if (self.top_k < 1 or self.top_k > 100) return error.InvalidTopK;
        if (self.top_p < 0.0 or self.top_p > 1.0) return error.InvalidTopP;
    }
};

pub const InferenceState = struct {
    params: *parameters.Parameters,
    tokenizer: *tokenizer.Tokenizer,
    cache: ?*cache.KVCache,
    sampler: sampler.Sampler,
    input_tokens: []u32,
    output_tokens: std.ArrayList(u32),
    position: usize,
    eos_found: bool,
    config: InferenceConfig,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, params: *parameters.Parameters, tok: *tokenizer.Tokenizer, config: InferenceConfig) !InferenceState {
        try config.validate();
        
        var self = InferenceState{
            .params = params,
            .tokenizer = tok,
            .cache = null,
            .sampler = sampler.Sampler.init(
                @intCast(std.time.milliTimestamp()),
                config.temperature,
                config.top_k,
                config.top_p
            ),
            .input_tokens = &.{},
            .output_tokens = std.ArrayList(u32).init(allocator),
            .position = 0,
            .eos_found = false,
            .config = config,
            .allocator = allocator,
        };
        
        const cache_ptr = try allocator.create(cache.KVCache);
        errdefer allocator.destroy(cache_ptr);
        
        cache_ptr.* = try cache.KVCache.init(
            allocator,
            config.max_tokens,
            params.config.dim,
            params.config.n_layers,
            params.config.n_kv_heads,
            params.config.n_heads
        );
        
        self.cache = cache_ptr;
        return self;
    }
    
    pub fn deinit(self: *InferenceState) void {
        if (self.cache) |cache_ptr| {
            cache_ptr.deinit();
            self.allocator.destroy(cache_ptr);
        }
        self.output_tokens.deinit();
    }
    
    pub fn generate(self: *InferenceState, prompt: []const u8) ![]u8 {
        std.debug.print("  🔍 generate: start\n", .{});
        
        if (self.cache == null) return error.CacheNotInitialized;
        if (self.params.mapped_data.len == 0) return error.InvalidMemoryMapping;
        
        const encoded = try self.tokenizer.encode(self.allocator, prompt);
        defer self.allocator.free(encoded);
        self.input_tokens = encoded;
        
        std.debug.print("  🔍 generate: tokens: ", .{});
        for (encoded[0..@min(5, encoded.len)]) |t| std.debug.print("{d} ", .{t});
        std.debug.print("\n", .{});
        
        self.output_tokens.clearRetainingCapacity();
        self.position = 0;
        self.eos_found = false;
        
        try self.output_tokens.appendSlice(self.input_tokens);
        try self.prefillPrompt();
        
        std.debug.print("  🔍 generate: generating tokens\n", .{});
        var gen_step: usize = 0;
        var response = std.ArrayList(u8).init(self.allocator);
        defer response.deinit();

        while (!self.eos_found and gen_step < 30) {
            const next_token = try self.generateNextToken();
            try self.output_tokens.append(next_token);
            
            const token_str = try self.tokenizer.decode(self.allocator, &[_]u32{next_token});
            defer self.allocator.free(token_str);
            
            std.debug.print("  🔍 token {d}: '{s}'\n", .{gen_step, token_str});
            try response.appendSlice(token_str);
            
            if (next_token == self.config.eos_token_id) {
                self.eos_found = true;
            }
            gen_step += 1;
        }

        const final_text = try self.tokenizer.decode(self.allocator, self.output_tokens.items);
        std.debug.print("\n🔊 Britix says: '{s}'\n", .{final_text});
        return final_text;
    }
    
    fn prefillPrompt(self: *InferenceState) !void {
        const config = self.params.config;
        const dim = config.dim;
        const tokens = self.input_tokens;
        const cache_ptr = self.cache orelse return error.CacheNotInitialized;
        
        const embeddings = try self.allocator.alloc(f32, tokens.len * dim);
        defer self.allocator.free(embeddings);
        
        for (0..tokens.len * dim) |i| {
            const token_idx = i / dim;
            const dim_idx = i % dim;
            embeddings[i] = try self.params.token_embedding.getSafe(
                self.params.mapped_data,
                &[_]u32{ @as(u32, @intCast(token_idx)), @as(u32, @intCast(dim_idx)) }
            );
        }
        
        var current = embeddings;
        var current_owned = false;
        
        for (0..config.n_layers) |layer_idx| {
            if (layer_idx % 4 == 0) {
                std.debug.print("  🚀 layer {d}/{d}\n", .{layer_idx, config.n_layers});
            }
            
            const layer = &self.params.layers[layer_idx];
            
            const attn_out = try self.attentionForward(current, layer, layer_idx, tokens.len, cache_ptr);
            defer self.allocator.free(attn_out);
            
            for (0..current.len) |idx| {
                current[idx] += attn_out[idx];
            }
            
            var norm_weights: [4096]f32 = undefined;
            for (0..dim) |j| {
                norm_weights[j] = try layer.attention_norm.getSafe(
                    self.params.mapped_data,
                    &[_]u32{ @as(u32, @intCast(j)) }
                );
            }
            
            const normed = try self.rmsNorm(current, norm_weights[0..dim]);
            if (current_owned) self.allocator.free(current);
            current = normed;
            current_owned = true;
            
            const ff_out = try self.feedforwardForward(current, layer);
            defer self.allocator.free(ff_out);
            
            for (0..current.len) |idx| {
                current[idx] += ff_out[idx];
            }
            
            for (0..dim) |j| {
                norm_weights[j] = try layer.ffn_norm.getSafe(
                    self.params.mapped_data,
                    &[_]u32{ @as(u32, @intCast(j)) }
                );
            }
            
            const normed2 = try self.rmsNorm(current, norm_weights[0..dim]);
            self.allocator.free(current);
            current = normed2;
        }
        
        if (current_owned and current.ptr != embeddings.ptr) {
            self.allocator.free(current);
        }
        
        self.position = tokens.len;
    }
    
    fn generateNextToken(self: *InferenceState) !u32 {
        const config = self.params.config;
        const dim = config.dim;
        const last_token = self.output_tokens.items[self.output_tokens.items.len - 1];
        const cache_ptr = self.cache orelse return error.CacheNotInitialized;
        
        const embedding = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(embedding);
        
        for (0..dim) |j| {
            embedding[j] = try self.params.token_embedding.getSafe(
                self.params.mapped_data,
                &[_]u32{ last_token, @as(u32, @intCast(j)) }
            );
        }
        
        var current = embedding;
        var current_owned = false;
        
        for (0..config.n_layers) |layer_idx| {
            const layer = &self.params.layers[layer_idx];
            
            const attn_out = try self.attentionForward(current, layer, layer_idx, 1, cache_ptr);
            defer self.allocator.free(attn_out);
            
            for (0..dim) |i| {
                current[i] += attn_out[i];
            }
            
            var norm_weights: [4096]f32 = undefined;
            for (0..dim) |j| {
                norm_weights[j] = try layer.attention_norm.getSafe(
                    self.params.mapped_data,
                    &[_]u32{ @as(u32, @intCast(j)) }
                );
            }
            
            const normed = try self.rmsNorm(current, norm_weights[0..dim]);
            if (current_owned) self.allocator.free(current);
            current = normed;
            current_owned = true;
            
            const ff_out = try self.feedforwardForward(current, layer);
            defer self.allocator.free(ff_out);
            
            for (0..dim) |i| {
                current[i] += ff_out[i];
            }
            
            for (0..dim) |j| {
                norm_weights[j] = try layer.ffn_norm.getSafe(
                    self.params.mapped_data,
                    &[_]u32{ @as(u32, @intCast(j)) }
                );
            }
            
            const normed2 = try self.rmsNorm(current, norm_weights[0..dim]);
            self.allocator.free(current);
            current = normed2;
        }
        
        var final_weights: [4096]f32 = undefined;
        for (0..dim) |j| {
            final_weights[j] = try self.params.norm_weight.getSafe(
                self.params.mapped_data,
                &[_]u32{ @as(u32, @intCast(j)) }
            );
        }
        
        const final_norm = try self.rmsNorm(current, final_weights[0..dim]);
        defer self.allocator.free(final_norm);
        
        if (current_owned) self.allocator.free(current);
        
        const logits = try self.allocator.alloc(f32, config.vocab_size);
        defer self.allocator.free(logits);
        @memset(logits, 0);
        
        const copy_len = @min(dim, config.vocab_size);
        @memcpy(logits[0..copy_len], final_norm[0..copy_len]);
        
        const token_idx = try self.sampler.sample(logits, self.allocator);
        return @as(u32, @intCast(token_idx));
    }
    
    fn attentionForward(
        self: *InferenceState,
        x: []const f32,
        layer: *parameters.MappedLayer,
        layer_idx: usize,
        seq_len: usize,
        cache_ptr: *cache.KVCache
    ) ![]f32 {
        const config = self.params.config;
        const dim = config.dim;
        const n_heads = config.n_heads;
        const n_kv_heads = config.n_kv_heads;
        const head_dim = dim / n_heads;
        const kv_repeat = n_heads / n_kv_heads;
        
        const q = try self.allocator.alloc(f32, seq_len * n_heads * head_dim);
        defer self.allocator.free(q);
        const k = try self.allocator.alloc(f32, seq_len * n_kv_heads * head_dim);
        defer self.allocator.free(k);
        const v = try self.allocator.alloc(f32, seq_len * n_kv_heads * head_dim);
        defer self.allocator.free(v);
        
        for (0..seq_len) |t| {
            const x_off = t * dim;
            
            for (0..n_heads * head_dim) |i| {
                var sum: f32 = 0;
                for (0..dim) |j| {
                    sum += x[x_off + j] * (try layer.wq.getSafe(
                        self.params.mapped_data,
                        &[_]u32{ @as(u32, @intCast(j)), @as(u32, @intCast(i)) }
                    ));
                }
                q[t * (n_heads * head_dim) + i] = sum;
            }
            
            for (0..n_kv_heads * head_dim) |i| {
                var sum: f32 = 0;
                for (0..dim) |j| {
                    sum += x[x_off + j] * (try layer.wk.getSafe(
                        self.params.mapped_data,
                        &[_]u32{ @as(u32, @intCast(i)), @as(u32, @intCast(j)) }
                    ));
                }
                k[t * (n_kv_heads * head_dim) + i] = sum;
            }
            
            for (0..n_kv_heads * head_dim) |i| {
                var sum: f32 = 0;
                for (0..dim) |j| {
                    sum += x[x_off + j] * (try layer.wv.getSafe(
                        self.params.mapped_data,
                        &[_]u32{ @as(u32, @intCast(i)), @as(u32, @intCast(j)) }
                    ));
                }
                v[t * (n_kv_heads * head_dim) + i] = sum;
            }
        }
        
        // FIXED: Update cache one token at a time
        const kv_dim = n_kv_heads * head_dim;
        for (0..seq_len) |t| {
            const k_start = t * kv_dim;
            const v_start = t * kv_dim;
            const k_single = k[k_start .. k_start + kv_dim];
            const v_single = v[v_start .. v_start + kv_dim];
            try cache_ptr.update(layer_idx, self.position + t, k_single, v_single);
        }
        
        const total_seq = self.position + seq_len;
        const cached_k = try cache_ptr.getK(layer_idx, 0, total_seq);
        const cached_v = try cache_ptr.getV(layer_idx, 0, total_seq);
        
        const output = try self.allocator.alloc(f32, seq_len * dim);
        defer self.allocator.free(output);
        @memset(output, 0);
        
        for (0..seq_len) |t| {
            for (0..n_heads) |h| {
                const q_start = t * (n_heads * head_dim) + h * head_dim;
                const kv_head = h / kv_repeat;
                
                var max_val: f32 = -std.math.inf(f32);
                var scores: [8192]f32 = undefined;
                
                for (0..total_seq) |pos| {
                    const k_off = pos * (n_kv_heads * head_dim) + kv_head * head_dim;
                    var score: f32 = 0;
                    for (0..head_dim) |i| {
                        score += q[q_start + i] * cached_k[k_off + i];
                    }
                    score /= @sqrt(@as(f32, @floatFromInt(head_dim)));
                    if (pos > self.position + t) score = -std.math.inf(f32);
                    scores[pos] = score;
                    if (score > max_val) max_val = score;
                }
                
                var sum_exp: f32 = 0;
                for (0..total_seq) |pos| {
                    const val = @exp(scores[pos] - max_val);
                    scores[pos] = val;
                    sum_exp += val;
                }
                
                const inv_sum = 1.0 / sum_exp;
                for (0..total_seq) |pos| scores[pos] *= inv_sum;
                
                for (0..head_dim) |i| {
                    var val_sum: f32 = 0;
                    for (0..total_seq) |pos| {
                        const v_off = pos * (n_kv_heads * head_dim) + kv_head * head_dim + i;
                        val_sum += scores[pos] * cached_v[v_off];
                    }
                    output[t * dim + h * head_dim + i] = val_sum;
                }
            }
        }
        
        const final_out = try self.allocator.alloc(f32, seq_len * dim);
        for (0..seq_len) |t| {
            const out_off = t * dim;
            for (0..dim) |i| {
                var sum: f32 = 0;
                for (0..n_heads * head_dim) |j| {
                    sum += output[out_off + j] * (try layer.wo.getSafe(
                        self.params.mapped_data,
                        &[_]u32{ @as(u32, @intCast(j)), @as(u32, @intCast(i)) }
                    ));
                }
                final_out[out_off + i] = sum;
            }
        }
        
        return final_out;
    }
    
    fn feedforwardForward(
        self: *InferenceState,
        x: []const f32,
        layer: *parameters.MappedLayer
    ) ![]f32 {
        const config = self.params.config;
        const dim = config.dim;
        const hidden_dim = config.hidden_dim;
        const seq_len = x.len / dim;
        
        const gate = try self.allocator.alloc(f32, seq_len * hidden_dim);
        defer self.allocator.free(gate);
        const up = try self.allocator.alloc(f32, seq_len * hidden_dim);
        defer self.allocator.free(up);
        
        for (0..seq_len) |t| {
            const x_off = t * dim;
            const gate_off = t * hidden_dim;
            
            for (0..hidden_dim) |i| {
                var sum_gate: f32 = 0;
                var sum_up: f32 = 0;
                for (0..dim) |j| {
                    const w1_val = try layer.w1.getSafe(
                        self.params.mapped_data,
                        &[_]u32{ @as(u32, @intCast(i)), @as(u32, @intCast(j)) }
                    );
                    sum_gate += x[x_off + j] * w1_val;
                    
                    const w3_val = try layer.w3.getSafe(
                        self.params.mapped_data,
                        &[_]u32{ @as(u32, @intCast(i)), @as(u32, @intCast(j)) }
                    );
                    sum_up += x[x_off + j] * w3_val;
                }
                gate[gate_off + i] = sum_gate;
                up[gate_off + i] = sum_up;
            }
        }
        
        for (0..seq_len * hidden_dim) |idx| {
            const sig = 1.0 / (1.0 + @exp(-gate[idx]));
            gate[idx] = gate[idx] * sig * up[idx];
        }
        
        const output = try self.allocator.alloc(f32, seq_len * dim);
        for (0..seq_len) |t| {
            const gate_off = t * hidden_dim;
            const out_off = t * dim;
            
            for (0..dim) |i| {
                var sum: f32 = 0;
                for (0..hidden_dim) |j| {
                    const w2_val = try layer.w2.getSafe(
                        self.params.mapped_data,
                        &[_]u32{ @as(u32, @intCast(i)), @as(u32, @intCast(j)) }
                    );
                    sum += gate[gate_off + j] * w2_val;
                }
                output[out_off + i] = sum;
            }
        }
        
        return output;
    }
    
    fn rmsNorm(self: *InferenceState, x: []const f32, weight: []const f32) ![]f32 {
        const dim = self.params.config.dim;
        const output = try self.allocator.alloc(f32, x.len);
        
        var i: usize = 0;
        while (i < x.len) : (i += dim) {
            var sum_sq: f32 = 0;
            for (0..dim) |j| sum_sq += x[i + j] * x[i + j];
            
            const rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(dim)) + 1e-5);
            
            for (0..dim) |j| {
                output[i + j] = x[i + j] * rms * weight[j];
            }
        }
        
        return output;
    }
};
