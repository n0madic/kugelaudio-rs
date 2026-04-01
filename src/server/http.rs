use std::io::Read as _;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use tiny_http::{Header, Method, Response, Server};

use super::protocol::{ErrorResponse, GenerateRequest, GenerateResponse, WorkItem};

/// Maximum allowed request body size (10 MB).
const MAX_BODY_BYTES: u64 = 10 * 1024 * 1024;
/// Timeout for waiting on a generation response from the server loop.
const REPLY_TIMEOUT: Duration = Duration::from_secs(300);

/// Spawn a thread that listens for HTTP requests.
///
/// Successful generation returns raw WAV bytes with `Content-Type: audio/wav`.
/// Errors return JSON with `Content-Type: application/json`.
///
/// This makes `curl -o output.wav http://host:port/generate -d '{"text":"..."}` just work.
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
                (Method::Post, "/generate") => handle_generate(&mut request, &work_tx),
                _ => json_response(&ErrorResponse::new("Not found".to_string()).to_json(), 404),
            };

            if let Err(e) = request.respond(response) {
                eprintln!("[http] Response error: {e}");
            }
        }
    })
}

fn handle_generate(
    request: &mut tiny_http::Request,
    work_tx: &mpsc::Sender<WorkItem>,
) -> Response<std::io::Cursor<Vec<u8>>> {
    // Reject oversized bodies before reading.
    let content_len = request.body_length().unwrap_or(0);
    if content_len > MAX_BODY_BYTES as usize {
        return error_response(
            &format!("Request body too large ({content_len} bytes, max {MAX_BODY_BYTES})"),
            413,
        );
    }

    let mut body = String::new();
    let read_result = request
        .as_reader()
        .take(MAX_BODY_BYTES + 1)
        .read_to_string(&mut body);

    if let Err(e) = read_result {
        return error_response(&format!("Failed to read body: {e}"), 400);
    }
    if body.len() as u64 > MAX_BODY_BYTES {
        return error_response("Request body too large", 413);
    }

    let req = match serde_json::from_str::<GenerateRequest>(&body) {
        Err(e) => return error_response(&format!("Invalid request JSON: {e}"), 400),
        Ok(r) => r,
    };

    let (reply_tx, reply_rx) = mpsc::channel();
    if work_tx
        .send(WorkItem {
            request: req,
            reply_tx,
        })
        .is_err()
    {
        return error_response("Server shutting down", 503);
    }

    let gen_response = reply_rx
        .recv_timeout(REPLY_TIMEOUT)
        .unwrap_or(GenerateResponse::Error("Generation timed out".to_string()));

    match gen_response {
        GenerateResponse::Ok(wav_bytes) => wav_response(wav_bytes),
        GenerateResponse::Error(msg) => error_response(&msg, 500),
    }
}

// ---------------------------------------------------------------------------
// Response builders
// ---------------------------------------------------------------------------

/// Build a raw WAV response with `Content-Type: audio/wav`.
fn wav_response(wav_bytes: Vec<u8>) -> Response<std::io::Cursor<Vec<u8>>> {
    let header = Header::from_bytes("Content-Type", "audio/wav").expect("static header is valid");
    Response::from_data(wav_bytes).with_header(header)
}

/// Build a JSON error response.
fn error_response(msg: &str, status_code: u32) -> Response<std::io::Cursor<Vec<u8>>> {
    json_response(&ErrorResponse::new(msg.to_string()).to_json(), status_code)
}

fn json_response(body: &str, status_code: u32) -> Response<std::io::Cursor<Vec<u8>>> {
    let header =
        Header::from_bytes("Content-Type", "application/json").expect("static header is valid");
    Response::from_string(body)
        .with_status_code(status_code)
        .with_header(header)
}

/// Expand `:port` shorthand to `0.0.0.0:port`.
fn normalize_bind_addr(addr: String) -> String {
    if addr.starts_with(':') {
        format!("0.0.0.0{addr}")
    } else {
        addr
    }
}
