// server.zig - BRITIX 8B HTTP SERVER - WITH FULL DEBUG
const std = @import("std");
const net = std.net;
const Britix = @import("britix.zig").BritixSystem;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("╔════════════════════════════════════════╗\n", .{});
    std.debug.print("║     BRITIX 8B HTTP SERVER v1.0        ║\n", .{});
    std.debug.print("║         PORT 8020 - INTERNET          ║\n", .{});
    std.debug.print("║  \"Formally Verified, Jolly Good!\"     ║\n", .{});
    std.debug.print("╚════════════════════════════════════════╝\n", .{});

    // Initialize Britix
    std.debug.print("\n[1/3] Loading Britix 8B...\n", .{});
    var britix = Britix.init(allocator) catch |err| {
        std.debug.print("❌ Failed: {}\n", .{err});
        return err;
    };
    defer britix.deinit();
    std.debug.print("✅ Britix ready\n", .{});

    // Create TCP server on port 8020
    std.debug.print("[2/3] Starting server on port 8020...\n", .{});
    const address = try net.Address.parseIp("0.0.0.0", 8020);
    var tcp_server = try address.listen(.{ 
        .reuse_port = true,
        .reuse_address = true 
    });
    defer tcp_server.deinit();

    std.debug.print("[3/3] Listening on http://localhost:8020\n", .{});
    std.debug.print("\n🇬🇧 Britix is ready, Governor! Tea at 3pm!\n", .{});
    std.debug.print("📡 Use: curl -X POST http://localhost:8020 -H \"Content-Type: application/json\" -d '{{\"prompt\":\"hello\"}}'\n\n", .{});

    var request_count: usize = 0;
    while (true) {
        std.debug.print("⏳ Waiting for connection...\n", .{});
        var client = tcp_server.accept() catch |err| {
            std.debug.print("❌ Accept error: {}\n", .{err});
            continue;
        };
        defer client.stream.close();
        
        request_count += 1;
        std.debug.print("📞 Connection #{d} accepted from {}\n", .{request_count, client.address});
        
        // Handle each client
        handleClient(allocator, &britix, client) catch |err| {
            std.debug.print("❌ Handle error: {}\n", .{err});
        };
    }
}

fn handleClient(allocator: std.mem.Allocator, britix: *Britix, client: net.Server.Connection) !void {
    std.debug.print("  🔍 handleClient: Starting\n", .{});
    
    var buffer: [8192]u8 = undefined;
    
    const request_line = client.stream.reader().readUntilDelimiter(&buffer, '\n') catch |err| {
        std.debug.print("  ❌ Failed to read request line: {}\n", .{err});
        return;
    };
    std.debug.print("  📝 Request line: {s}", .{request_line});
    
    if (std.mem.startsWith(u8, request_line, "POST")) {
        std.debug.print("  ✅ POST request detected\n", .{});
        var content_length: usize = 0;
        
        // Read headers
        std.debug.print("  📋 Reading headers...\n", .{});
        while (true) {
            const line = client.stream.reader().readUntilDelimiter(&buffer, '\n') catch |err| {
                if (err == error.EndOfStream) break;
                std.debug.print("  ❌ Header read error: {}\n", .{err});
                return;
            };
            
            if (line.len == 0 or (line.len == 1 and line[0] == '\r')) {
                std.debug.print("  ✅ End of headers\n", .{});
                break;
            }
            
            std.debug.print("  Header: {s}", .{line});
            
            if (std.mem.indexOf(u8, line, "Content-Length:")) |idx| {
                const val_str = std.mem.trim(u8, line[idx+15..], " \r");
                content_length = std.fmt.parseInt(usize, val_str, 10) catch {
                    std.debug.print("  ❌ Invalid Content-Length\n", .{});
                    return;
                };
                std.debug.print("  📦 Content-Length: {d}\n", .{content_length});
            }
        }

        if (content_length == 0 or content_length > 1024 * 1024) {
            std.debug.print("  ❌ Invalid content length: {d}\n", .{content_length});
            try sendJsonError(allocator, client, 400, "Invalid content length");
            return;
        }

        std.debug.print("  📥 Reading body ({d} bytes)...\n", .{content_length});
        const body = client.stream.reader().readAllAlloc(allocator, content_length) catch |err| {
            std.debug.print("  ❌ Failed to read body: {}\n", .{err});
            return;
        };
        defer allocator.free(body);
        std.debug.print("  📥 Body: {s}\n", .{body});

        std.debug.print("  🔍 Parsing JSON...\n", .{});
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch |err| {
            std.debug.print("  ❌ JSON parse error: {}\n", .{err});
            try sendJsonError(allocator, client, 400, "Invalid JSON");
            return;
        };
        defer parsed.deinit();

        const prompt_obj = parsed.value.object.get("prompt") orelse {
            std.debug.print("  ❌ Missing 'prompt' field\n", .{});
            try sendJsonError(allocator, client, 400, "Missing 'prompt' field");
            return;
        };

        const prompt = prompt_obj.string;
        std.debug.print("  💬 Prompt: '{s}'\n", .{prompt});

        std.debug.print("  🤔 Calling britix.generate()...\n", .{});
        const result = britix.generate(prompt) catch |err| {
            std.debug.print("  ❌ Generation error: {}\n", .{err});
            try sendJsonError(allocator, client, 500, "Generation failed");
            return;
        };
        defer allocator.free(result);
        std.debug.print("  ✅ Generation complete, result length: {d}\n", .{result.len});
        std.debug.print("  💬 Result: '{s}'\n", .{result});

        try sendJsonResponse(allocator, client, 200, result);
        std.debug.print("  ✅ Response sent to client\n", .{});
        
    } else {
        std.debug.print("  ℹ️  Sending info page (not POST)\n", .{});
        const response = 
            \\HTTP/1.1 200 OK
            \\Content-Type: text/plain
            \\Access-Control-Allow-Origin: *
            \\
            \\🇬🇧 Britix 8B API Server
            \\Port: 8020
            \\Usage: POST / with JSON {"prompt": "your text"}
            \\
            \\Example:
            \\curl -X POST http://localhost:8020 -H "Content-Type: application/json" -d '{"prompt":"hello"}'
            \\
        ;
        const bytes_written = client.stream.writer().write(response) catch |err| {
            std.debug.print("  ❌ Failed to write response: {}\n", .{err});
            return;
        };
        std.debug.print("  ✅ Info page sent, {d} bytes\n", .{bytes_written});
    }
    std.debug.print("  ✅ handleClient complete\n", .{});
}

fn sendJsonResponse(allocator: std.mem.Allocator, client: net.Server.Connection, status: u16, text: []const u8) !void {
    std.debug.print("  📤 sendJsonResponse: status={d}, text='{s}'\n", .{status, text});
    const writer = client.stream.writer();
    
    var escaped = std.ArrayList(u8).init(allocator);
    defer escaped.deinit();
    
    for (text) |c| {
        switch (c) {
            '"' => try escaped.appendSlice("\\\""),
            '\\' => try escaped.appendSlice("\\\\"),
            '\n' => try escaped.appendSlice("\\n"),
            '\r' => try escaped.appendSlice("\\r"),
            '\t' => try escaped.appendSlice("\\t"),
            else => try escaped.append(c),
        }
    }
    std.debug.print("  📤 Escaped text length: {d}\n", .{escaped.items.len});

    const status_text = if (status == 200) "OK" else "Bad Request";
    const json_response = try std.fmt.allocPrint(allocator, "{{\"response\":\"{s}\"}}", .{escaped.items});
    defer allocator.free(json_response);
    std.debug.print("  📤 JSON response: {s}\n", .{json_response});

    const response = try std.fmt.allocPrint(allocator,
        \\HTTP/1.1 {d} {s}
        \\Content-Type: application/json
        \\Content-Length: {d}
        \\Access-Control-Allow-Origin: *
        \\
        \\{s}
    , .{ status, status_text, json_response.len, json_response });
    defer allocator.free(response);
    std.debug.print("  📤 Full response length: {d}\n", .{response.len});
    
    const bytes_written = writer.write(response) catch |err| {
        std.debug.print("  ❌ Failed to write JSON response: {}\n", .{err});
        return;
    };
    std.debug.print("  ✅ JSON response sent, {d} bytes\n", .{bytes_written});
}

fn sendJsonError(allocator: std.mem.Allocator, client: net.Server.Connection, status: u16, message: []const u8) !void {
    std.debug.print("  ❌ sendJsonError: status={d}, message='{s}'\n", .{status, message});
    const writer = client.stream.writer();
    const status_text = if (status == 400) "Bad Request" else "Error";
    
    const json_response = try std.fmt.allocPrint(allocator, "{{\"error\":\"{s}\"}}", .{message});
    defer allocator.free(json_response);

    const response = try std.fmt.allocPrint(allocator,
        \\HTTP/1.1 {d} {s}
        \\Content-Type: application/json
        \\Content-Length: {d}
        \\Access-Control-Allow-Origin: *
        \\
        \\{s}
    , .{ status, status_text, json_response.len, json_response });
    defer allocator.free(response);
    
    _ = writer.write(response) catch {};
    std.debug.print("  ✅ Error response sent\n", .{});
}
