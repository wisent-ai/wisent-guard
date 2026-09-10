//! Writing back: one JSON document for a plain call, or the NDJSON event
//! stream a streamed job reports its logs and result through.

use std::io::Write;

use serde_json::{json, Value};

use super::SharedWriter;

// MARK: - Responses

pub(super) fn send_json(writer: &SharedWriter, status: u16, document: Value) {
    let body = serde_json::to_string_pretty(&document).unwrap_or_default();
    write_response_head(writer, status, "application/json", Some(body.len()));
    write_bytes(writer, body.as_bytes());
}

pub(super) fn send_error(writer: &SharedWriter, status: u16, message: &str) {
    send_json(writer, status, json!({"error": message}));
}

pub(super) fn emit_log(writer: &SharedWriter, stream: &str, chunk: &str) {
    emit_event(writer, &json!({"type": "log", "stream": stream, "chunk": chunk}));
}

pub(super) fn emit_result(writer: &SharedWriter, status: i32, document: Value) {
    emit_event(writer, &json!({"type": "result", "status": status, "json": document}));
}

pub(super) fn emit_event(writer: &SharedWriter, event: &Value) {
    let mut line = serde_json::to_string(event).unwrap_or_default();
    line.push('\n');
    write_bytes(writer, line.as_bytes());
}

pub(super) fn write_response_head(
    writer: &SharedWriter,
    status: u16,
    content_type: &str,
    content_length: Option<usize>,
) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let mut head =
        format!("HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\nconnection: close\r\n");
    match content_length {
        Some(length) => head.push_str(&format!("content-length: {length}\r\n\r\n")),
        None => head.push_str("cache-control: no-cache\r\n\r\n"),
    }
    write_bytes(writer, head.as_bytes());
}

pub(super) fn write_bytes(writer: &SharedWriter, bytes: &[u8]) {
    if let Ok(mut stream) = writer.lock() {
        let _ = stream.write_all(bytes);
        let _ = stream.flush();
    }
}
