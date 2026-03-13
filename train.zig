// train.zig - Training loop for Britix 8B
// "Teaching an old droid new tricks. Jolly good fun!"

const std = @import("std");
const builtin = @import("builtin");

const parameters = @import("parameters.zig");
const logger = @import("logger.zig");

pub const TrainingConfig = struct {
    // Optimizer settings
    learning_rate: f32 = 3e-4,
    weight_decay: f32 = 0.1,
    beta1: f32 = 0.9,
    beta2: f32 = 0.95,
    
    // Training settings
    batch_size: usize = 32,
    micro_batch_size: usize = 4,
    gradient_accumulation_steps: usize = 8,
    max_steps: usize = 100_000,
    warmup_steps: usize = 2_000,
    
    // Regularization
    dropout: f32 = 0.0,
    label_smoothing: f32 = 0.0,
    
    // Logging & checkpointing
    log_interval: usize = 10,
    eval_interval: usize = 500,
    save_interval: usize = 1000,
    eval_steps: usize = 100,
    
    // Precision
    mixed_precision: bool = true,
    gradient_clip: f32 = 1.0,
    
    pub fn validate(self: *const TrainingConfig) !void {
        if (self.learning_rate <= 0) return error.InvalidLearningRate;
        if (self.batch_size < self.micro_batch_size) return error.InvalidBatchSize;
        if (self.gradient_accumulation_steps * self.micro_batch_size != self.batch_size) 
            return error.InvalidGradientAccumulation;
    }
};

pub const Trainer = struct {
    const Self = @This();
    
    params: *parameters.Parameters,
    config: TrainingConfig,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, params: *parameters.Parameters, config: TrainingConfig) !Self {
        try config.validate();
        
        return Self{
            .params = params,
            .config = config,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        _ = self;
    }
    
    fn get_lr(self: *Self) f32 {
        if (self.config.warmup_steps == 0) return self.config.learning_rate;
        if (self.config.max_steps < self.config.warmup_steps) return self.config.learning_rate;
        return self.config.learning_rate;
    }
};

test "trainer - config validation" {
    const good_config = TrainingConfig{};
    try good_config.validate();
    
    const bad_config = TrainingConfig{ .learning_rate = -1.0 };
    try std.testing.expectError(error.InvalidLearningRate, bad_config.validate());
    
    const bad_config2 = TrainingConfig{ 
        .batch_size = 4,
        .micro_batch_size = 8,
    };
    try std.testing.expectError(error.InvalidBatchSize, bad_config2.validate());
}

test "trainer - learning rate calculation" {
    const allocator = std.testing.allocator;
    
    const config = parameters.BritixConfig{
        .dim = 2048,
        .n_layers = 36,
        .n_heads = 16,
        .n_kv_heads = 2,
        .vocab_size = 151936,
        .hidden_dim = 5632,
        .max_seq_len = 32768,
        .norm_eps = 1e-6,
        .rope_theta = 1000000.0,
    };
    
    const mock_data = try allocator.alloc(u8, 1024);
    defer allocator.free(mock_data);
    @memset(mock_data, 0);
    
    var params = parameters.Parameters{
        .config = config,
        .mapped_data = mock_data,
        .token_embedding = undefined,
        .layers = undefined,
        .norm_weight = undefined,
    };
    
    // FIXED: Removed the ", "test_norm" string parameter
    params.token_embedding = parameters.MappedTensor.init(0, &[_]u32{ config.vocab_size, config.dim });
    params.norm_weight = parameters.MappedTensor.init(0, &[_]u32{config.dim});
    
    params.layers = try allocator.alloc(parameters.MappedLayer, 1);
    defer allocator.free(params.layers);
    
    var offset: u64 = 0;
    params.layers[0] = parameters.MappedLayer.init(&offset, config, 0);
    
    const train_config = TrainingConfig{};
    var trainer = try Trainer.init(allocator, &params, train_config);
    defer trainer.deinit();
    
    const lr = trainer.get_lr();
    try std.testing.expect(lr > 0);
}
