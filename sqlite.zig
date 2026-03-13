// sqlite.zig - SQLite binding for Britix memory
// "Perfect recall, like a British butler's mind."

const std = @import("std");

pub const Db = struct {
    handle: ?*c.sqlite3,
    allocator: std.mem.Allocator,

    pub const Mode = enum {
        File,
        Memory,
    };

    pub const OpenFlags = struct {
        create: bool = false,
        read_write: bool = true,
    };

    pub fn init(options: struct {
        mode: Mode,
        flags: OpenFlags,
    }) !Db {
        _ = options;
        return Db{
            .handle = null,
            .allocator = std.heap.page_allocator,
        };
    }

    pub fn open(self: *Db, path: []const u8) !void {
        var handle: ?*c.sqlite3 = null;
        const flags = c.SQLITE_OPEN_READWRITE | c.SQLITE_OPEN_CREATE;
        
        // Convert path to C string
        const path_c = try self.allocator.dupeZ(u8, path);
        defer self.allocator.free(path_c);
        
        if (c.sqlite3_open_v2(path_c, &handle, flags, null) != c.SQLITE_OK) {
            return error.SqliteOpenFailed;
        }
        self.handle = handle;
    }

    pub fn exec(self: *Db, sql: []const u8, params: anytype) !void {
        _ = params;
        var errmsg: [*c]u8 = null;
        
        // Convert sql to C string
        const sql_c = try self.allocator.dupeZ(u8, sql);
        defer self.allocator.free(sql_c);
        
        if (c.sqlite3_exec(self.handle, sql_c, null, null, &errmsg) != c.SQLITE_OK) {
            if (errmsg) |msg| {
                std.debug.print("SQL error: {s}\n", .{msg});
                c.sqlite3_free(msg);
            }
            return error.SqliteExecFailed;
        }
    }

    pub fn prepare(self: *Db, sql: []const u8) !Statement {
        var stmt: ?*c.sqlite3_stmt = null;
        
        // Convert sql to C string
        const sql_c = try self.allocator.dupeZ(u8, sql);
        defer self.allocator.free(sql_c);
        
        if (c.sqlite3_prepare_v2(self.handle, sql_c, -1, &stmt, null) != c.SQLITE_OK) {
            return error.SqlitePrepareFailed;
        }
        return Statement{
            .handle = stmt,
            .allocator = self.allocator,
        };
    }

    pub fn deinit(self: *Db) void {
        if (self.handle) |h| {
            _ = c.sqlite3_close(h);
        }
    }
};

pub const Statement = struct {
    handle: ?*c.sqlite3_stmt,
    allocator: std.mem.Allocator,

    pub fn bind(self: *Statement, args: anytype) !void {
        const args_type = @TypeOf(args);
        const args_info = @typeInfo(args_type);

        if (args_info == .Struct) {
            const fields = args_info.Struct.fields;
            inline for (fields, 0..) |field, i| {
                const value = @field(args, field.name);
                const idx = @as(c_int, @intCast(i + 1));
                switch (@TypeOf(value)) {
                    i64 => {
                        if (c.sqlite3_bind_int64(self.handle, idx, value) != c.SQLITE_OK) {
                            return error.SqliteBindFailed;
                        }
                    },
                    f64 => {
                        if (c.sqlite3_bind_double(self.handle, idx, value) != c.SQLITE_OK) {
                            return error.SqliteBindFailed;
                        }
                    },
                    []const u8 => {
                        if (c.sqlite3_bind_text(self.handle, idx, value.ptr, @intCast(value.len), c.SQLITE_STATIC) != c.SQLITE_OK) {
                            return error.SqliteBindFailed;
                        }
                    },
                    else => @compileError("Unsupported type for SQLite bind"),
                }
            }
        }
    }

    pub fn step(self: *Statement) !bool {
        const rc = c.sqlite3_step(self.handle);
        switch (rc) {
            c.SQLITE_ROW => return true,
            c.SQLITE_DONE => return false,
            else => return error.SqliteStepFailed,
        }
    }

    pub fn int(self: *Statement, col: u32) !i64 {
        return c.sqlite3_column_int64(self.handle, @intCast(col));
    }

    pub fn float(self: *Statement, col: u32) !f64 {
        return c.sqlite3_column_double(self.handle, @intCast(col));
    }

    pub fn text(self: *Statement, col: u32) ![]const u8 {
        const ptr = c.sqlite3_column_text(self.handle, @intCast(col));
        const len = c.sqlite3_column_bytes(self.handle, @intCast(col));
        return ptr[0..@intCast(len)];
    }

    pub fn deinit(self: *Statement) void {
        if (self.handle) |h| {
            _ = c.sqlite3_finalize(h);
        }
    }
};

// C declarations with proper calling convention
pub const c = struct {
    pub const sqlite3 = opaque {};
    pub const sqlite3_stmt = opaque {};

    pub const SQLITE_OK = 0;
    pub const SQLITE_ROW = 100;
    pub const SQLITE_DONE = 101;
    pub const SQLITE_OPEN_READWRITE = 2;
    pub const SQLITE_OPEN_CREATE = 4;
    pub const SQLITE_STATIC = @as(?*const fn (ptr: ?*anyopaque) callconv(.C) void, @ptrFromInt(0));

    extern fn sqlite3_open_v2(filename: [*c]const u8, ppDb: [*c]?*sqlite3, flags: c_int, zVfs: [*c]const u8) c_int;
    extern fn sqlite3_close(db: ?*sqlite3) c_int;
    extern fn sqlite3_exec(db: ?*sqlite3, sql: [*c]const u8, callback: ?*anyopaque, arg: ?*anyopaque, errmsg: [*c][*c]u8) c_int;
    extern fn sqlite3_prepare_v2(db: ?*sqlite3, zSql: [*c]const u8, nByte: c_int, ppStmt: [*c]?*sqlite3_stmt, pzTail: [*c][*c]const u8) c_int;
    extern fn sqlite3_finalize(pStmt: ?*sqlite3_stmt) c_int;
    extern fn sqlite3_step(pStmt: ?*sqlite3_stmt) c_int;
    extern fn sqlite3_bind_int64(pStmt: ?*sqlite3_stmt, i: c_int, value: i64) c_int;
    extern fn sqlite3_bind_double(pStmt: ?*sqlite3_stmt, i: c_int, value: f64) c_int;
    extern fn sqlite3_bind_text(pStmt: ?*sqlite3_stmt, i: c_int, value: [*c]const u8, n: c_int, destructor: ?*const fn (ptr: ?*anyopaque) callconv(.C) void) c_int;
    extern fn sqlite3_column_int64(pStmt: ?*sqlite3_stmt, iCol: c_int) i64;
    extern fn sqlite3_column_double(pStmt: ?*sqlite3_stmt, iCol: c_int) f64;
    extern fn sqlite3_column_text(pStmt: ?*sqlite3_stmt, iCol: c_int) [*c]const u8;
    extern fn sqlite3_column_bytes(pStmt: ?*sqlite3_stmt, iCol: c_int) c_int;
    extern fn sqlite3_free(ptr: ?*anyopaque) void;
};
