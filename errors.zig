// errors.zig - Britix error definitions
// "Oh dear, an error! But we're British, so we'll handle it gracefully."

const std = @import("std");

pub const BritixError = error{
    // Memory errors
    OutOfMemory,
    MemoryLeak,
    InvalidAllocator,
    
    // Tensor errors
    DimensionMismatch,
    IndexOutOfBounds,
    InvalidShape,
    TensorNotFound,
    
    // Model errors
    ModelNotFound,
    InvalidConfig,
    UnsupportedVersion,
    CorruptedWeights,
    
    // Inference errors
    ContextOverflow,
    InvalidToken,
    SamplingFailed,
    
    // IO errors
    FileNotFound,
    PermissionDenied,
    InvalidFormat,
    
    // GPU errors
    CudaError,
    GpuUnavailable,
    KernelLaunchFailed,
    
    // Ethical errors
    UnsafeAction,
    BritishPolitenessViolation,
};

pub fn formatError(err: BritixError) []const u8 {
    return switch (err) {
        BritixError.OutOfMemory => "I say, we appear to be out of memory! How dreadfully embarrassing.",
        BritixError.MemoryLeak => "Blast! A memory leak! The villain strikes again!",
        BritixError.DimensionMismatch => "These tensors don't match! Like socks with sandals, what!",
        BritixError.ModelNotFound => "Can't find the model, old boy. Have you checked under the tea cozy?",
        BritixError.PermissionDenied => "I'm terribly sorry, but I simply cannot allow that. It's just not cricket.",
        BritixError.BritishPolitenessViolation => "I say! That was rather rude. We don't do that here.",
        else => "Oh dear, something went wrong. Let's have a cup of tea and figure it out.",
    };
}

pub fn handleError(err: anyerror) void {
    std.debug.print("❌ ", .{});
    if (@hasDecl(@TypeOf(err), "description")) {
        std.debug.print("{s}\n", .{@errorName(err)});
    } else {
        std.debug.print("Unexpected error: {}\n", .{err});
    }
    std.debug.print("🇬🇧 Recommended action: Have a cup of Earl Grey and try again.\n", .{});
}

test "error formatting" {
    const err = BritixError.DimensionMismatch;
    const msg = formatError(err);
    try std.testing.expect(msg.len > 0);
}
