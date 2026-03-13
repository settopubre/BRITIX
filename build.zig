// build.zig - Master build file for Britix
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Server executable
    const server = b.addExecutable(.{
        .name = "server",
        .root_source_file = b.path("server.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Link SQLite
    server.linkSystemLibrary("sqlite3");
    server.linkLibC();
    
    b.installArtifact(server);

    // Run server
    const run_cmd = b.addRunArtifact(server);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    
    const run_step = b.step("run", "Run the server");
    run_step.dependOn(&run_cmd.step);
}
