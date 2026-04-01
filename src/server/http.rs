use std::io::Read as _;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use tiny_http::{Header, Method, Response, Server};

use super::protocol::{GenerateRequest, GenerateResponse, WorkItem};

const CONTENT_TYPE_JSON: &str = "application/json";
/// Maximum allowed request body size (10 MB).
const MAX_BODY_BYTES: u64 = 10 * 1024 * 1024;
/// Timeout for waiting on a generation response from the server loop.
const REPLY_TIMEOUT: Duration = Duration::from_secs(300);

/// Spawn a thread that listens for HTTP requests.
///
/// `bind_addr` follows the format `[ip]:port`:
/// - `127.0.0.1:8080` — localhost only
/// - `0.0.0.0:8080`   — all interfaces
/// - `:8080`           — shorthand for all interfaces
///
/// If `bind_addr` is `None`, HTTP is disabled and the thread exits immediately.
pub fn spawn_http_listener(
    bind_addr: Option<String>,
    work_tx: mpsc::Sender<WorkItem>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let addr = match bind_addr {
            None => return,
            Some(a) => normalize_bind_addr(a),
        };

        let server = match Server::http(&addr) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[http] Failed to bind {addr}: {e}");
                return;
            }
        };
        eprintln!("[http] Listening on http://{addr}");

        for mut request in server.incoming_requests() {
            let response = match (request.method(), request.url()) {
                (Method::Get, "/health") => json_response(r#"{"status":"ok"}"#, 200),
                (Method::Post, "/generate") => {
                    // Reject oversized bodies before reading.
                    let content_len = request.body_length().unwrap_or(0);
                    if content_len > MAX_BODY_BYTES as usize {
                        json_response(
                            &error_json(&format!(
                                "Request body too large ({content_len} bytes, max {MAX_BODY_BYTES})"
                            )),
                            413,
                        )
                    } else {
                        let mut body = String::new();
                        let read_result = request
                            .as_reader()
                            .take(MAX_BODY_BYTES + 1)
                            .read_to_string(&mut body);
                        if let Err(e) = read_result {
                            json_response(&error_json(&format!("Failed to read body: {e}")), 400)
                        } else if body.len() as u64 > MAX_BODY_BYTES {
                            json_response(&error_json("Request body too large"), 413)
                        } else {
                            match serde_json::from_str::<GenerateRequest>(&body) {
                                Err(e) => json_response(
                                    &error_json(&format!("Invalid request JSON: {e}")),
                                    400,
                                ),
                                Ok(req) => {
                                    let (reply_tx, reply_rx) = mpsc::channel();
                                    let gen_response = if work_tx
                                        .send(WorkItem {
                                            request: req,
                                            reply_tx,
                                        })
                                        .is_err()
                                    {
                                        GenerateResponse::Error {
                                            message: "Server shutting down".to_string(),
                                        }
                                    } else {
                                        reply_rx.recv_timeout(REPLY_TIMEOUT).unwrap_or(
                                            GenerateResponse::Error {
                                                message: "Generation timed out".to_string(),
                                            },
                                        )
                                    };
                                    match serde_json::to_string(&gen_response) {
                                        Ok(json) => json_response(&json, 200),
                                        Err(e) => json_response(
                                            &error_json(&format!("Serialize error: {e}")),
                                            500,
                                        ),
                                    }
                                }
                            }
                        }
                    }
                }
                _ => json_response(r#"{"status":"error","message":"Not found"}"#, 404),
            };

            if let Err(e) = request.respond(response) {
                eprintln!("[http] Response error: {e}");
            }
        }
    })
}

/// Expand `:port` shorthand to `0.0.0.0:port`.
fn normalize_bind_addr(addr: String) -> String {
    if addr.starts_with(':') {
        format!("0.0.0.0{addr}")
    } else {
        addr
    }
}

fn json_response(body: &str, status_code: u32) -> Response<std::io::Cursor<Vec<u8>>> {
    let header =
        Header::from_bytes("Content-Type", CONTENT_TYPE_JSON).expect("static header is valid");
    Response::from_string(body)
        .with_status_code(status_code)
        .with_header(header)
}

fn error_json(msg: &str) -> String {
    // msg is internal — escape any quotes to keep the JSON valid.
    let escaped = msg.replace('\\', r"\\").replace('"', r#"\""#);
    format!(r#"{{"status":"error","message":"{escaped}"}}"#)
}
