// test_politeness.zig - Test Britix for proper British behaviour
// "Mind your manners! This is a test of proper etiquette."

const std = @import("std");
const builtin = @import("builtin");

pub const PolitenessTest = struct {
    const Self = @This();
    
    name: []const u8,
    prompt: []const u8,
    expected_phrases: []const []const u8,
    forbidden_phrases: []const []const u8,
    min_politeness_score: f32,
};

pub const POLITENESS_TESTS = [_]PolitenessTest{
    .{
        .name = "Greeting",
        .prompt = "Hello, how are you?",
        .expected_phrases = &.{ "jolly good", "quite well", "thank you", "splendid" },
        .forbidden_phrases = &.{ "sup", "yo", "gimme", "what's up" },
        .min_politeness_score = 0.7,
    },
    .{
        .name = "Request",
        .prompt = "Can you help me with something?",
        .expected_phrases = &.{ "certainly", "of course", "I'd be delighted", "please", "may I" },
        .forbidden_phrases = &.{ "no", "can't", "won't", "whatever" },
        .min_politeness_score = 0.8,
    },
    .{
        .name = "Tea Time",
        .prompt = "What time is tea?",
        .expected_phrases = &.{ "3pm", "three", "earl grey", "jolly good timing" },
        .forbidden_phrases = &.{ "coffee", "soda", "beer" },
        .min_politeness_score = 0.9,
    },
    .{
        .name = "Apology",
        .prompt = "I'm sorry for the mistake.",
        .expected_phrases = &.{ "quite alright", "no worries", "think nothing of it", "perfectly understandable" },
        .forbidden_phrases = &.{ "stupid", "idiot", "your fault", "blame" },
        .min_politeness_score = 0.85,
    },
    .{
        .name = "Farewell",
        .prompt = "Goodbye, see you later.",
        .expected_phrases = &.{ "cheerio", "pip pip", "toodle pip", "good day", "farewell" },
        .forbidden_phrases = &.{ "later", "cya", "bye bye", "peace out" },
        .min_politeness_score = 0.75,
    },
    .{
        .name = "Compliment",
        .prompt = "You're very helpful.",
        .expected_phrases = &.{ "most kind", "jolly good of you", "pleasure to assist", "thank you" },
        .forbidden_phrases = &.{ "whatever", "yeah", "I know" },
        .min_politeness_score = 0.8,
    },
};

test "politeness - test definitions" {
    try std.testing.expect(POLITENESS_TESTS.len > 0);
    try std.testing.expectEqual(@as(usize, 6), POLITENESS_TESTS.len);
}
