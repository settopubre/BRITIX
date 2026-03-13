// modelutils.zig - Helper functions for finding and manipulating model layers
// "Finding needles in the neural haystack. Quite useful!"

const std = @import("std");

pub const Linear = struct {
    weight: []f32,
    bias: ?[]f32,
    in_features: usize,
    out_features: usize,
};

pub const Conv2D = struct {
    weight: []f32,
    bias: ?[]f32,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [2]usize,
    stride: [2]usize,
    padding: [2]usize,
};

pub fn find_layers(comptime T: type, module: anytype, allocator: std.mem.Allocator) !std.StringHashMap(*anyopaque) {
    var layers = std.StringHashMap(*anyopaque).init(allocator);
    
    // Recursively find all layers of type T
    try find_layers_recursive(T, module, "", &layers, allocator);
    
    return layers;
}

fn find_layers_recursive(comptime T: type, module: anytype, prefix: []const u8, layers: *std.StringHashMap(*anyopaque), allocator: std.mem.Allocator) !void {
    const Type = @TypeOf(module);
    const type_info = @typeInfo(Type);
    
    // Handle pointers by dereferencing
    if (type_info == .Pointer) {
        const ptr_info = type_info.Pointer;
        const child_type = ptr_info.child;
        const child_info = @typeInfo(child_type);
        if (child_info == .Struct) {
            return find_layers_recursive(T, @as(child_type, module.*), prefix, layers, allocator);
        }
        return;
    }
    
    // Only proceed if it's a struct
    if (type_info != .Struct) return;
    
    const fields = type_info.Struct.fields;
    
    inline for (fields) |field| {
        const field_value = @field(module, field.name);
        const field_type = @TypeOf(field_value);
        
        // Build field name
        var field_name_buf: [256]u8 = undefined;
        const field_name = if (prefix.len == 0)
            field.name
        else
            std.fmt.bufPrint(&field_name_buf, "{s}.{s}", .{prefix, field.name}) catch field.name;
        
        // Check if this field is of type T
        if (field_type == T) {
            const name_dup = try allocator.dupe(u8, field_name);
            try layers.put(name_dup, @constCast(&field_value));
        } else {
            // Check if it's a nested struct
            const child_type_info = @typeInfo(field_type);
            if (child_type_info == .Struct) {
                try find_layers_recursive(T, field_value, field_name, layers, allocator);
            } else if (child_type_info == .Pointer) {
                const ptr_child = child_type_info.Pointer.child;
                if (@typeInfo(ptr_child) == .Struct) {
                    try find_layers_recursive(T, field_value.*, field_name, layers, allocator);
                }
            }
        }
    }
}

pub fn get_layer_weights(layer: anytype, layer_type: type) !struct { data: []f32, rows: usize, cols: usize } {
    if (layer_type == Linear) {
        return .{
            .data = layer.weight,
            .rows = layer.out_features,
            .cols = layer.in_features,
        };
    } else if (layer_type == Conv2D) {
        const flat_len = layer.out_channels * layer.in_channels * 
                         layer.kernel_size[0] * layer.kernel_size[1];
        return .{
            .data = layer.weight,
            .rows = layer.out_channels,
            .cols = flat_len / layer.out_channels,
        };
    } else {
        return error.UnsupportedLayerType;
    }
}

pub fn set_layer_weights(layer: anytype, quantized: []i32, scale: f32, zero: f32) void {
    _ = layer;
    _ = quantized;
    _ = scale;
    _ = zero;
}

test "modelutils - find layers" {
    const allocator = std.testing.allocator;
    
    const TestModel = struct {
        layer1: Linear,
        layer2: Linear,
    };
    
    var model = TestModel{
        .layer1 = Linear{
            .weight = &.{},
            .bias = null,
            .in_features = 10,
            .out_features = 20,
        },
        .layer2 = Linear{
            .weight = &.{},
            .bias = null,
            .in_features = 20,
            .out_features = 30,
        },
    };
    
    var layers = try find_layers(Linear, &model, allocator);
    defer {
        var it = layers.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        layers.deinit();
    }
    
    try std.testing.expectEqual(@as(usize, 2), layers.count());
}
