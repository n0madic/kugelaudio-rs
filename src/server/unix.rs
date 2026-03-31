use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::sync::mpsc;
use std::thread;

use super::protocol::{GenerateRequest, GenerateResponse, WorkItem};

/// Spawn a thread that listens on a Unix domain socket.
///
/// Each accepted connection is handled in its own thread:
/// - Read one JSON line → dispatch to main loop → write one JSON response line.
///
/// A stale socket file from a previous run is removed before binding.
pub fn spawn_unix_listener(
    socket_path: String,
    work_tx: mpsc::Sender<WorkItem>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // Remove stale socket from a previous run.
        let _ = std::fs::remove_file(&socket_path);

        let listener = match UnixListener::bind(&socket_path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[unix] Failed to bind {socket_path}: {e}");
                return;
            }
        };
        eprintln!("[unix] Listening on {socket_path}");

        for stream in listener.incoming() {
            match stream {
                Err(e) => eprintln!("[unix] Accept error: {e}"),
                Ok(stream) => {
                    let work_tx = work_tx.clone();
                    thread::spawn(move || handle_connection(stream, work_tx));
                }
            }
        }
    })
}

fn handle_connection(stream: std::os::unix::net::UnixStream, work_tx: mpsc::Sender<WorkItem>) {
    let mut write_half = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[unix] Stream clone error: {e}");
            return;
        }
    };

    let mut reader = BufReader::new(stream);
    let mut line = String::new();

    if let Err(e) = reader.read_line(&mut line) {
        eprintln!("[unix] Read error: {e}");
        return;
    }

    let trimmed = line.trim();
    if trimmed.is_empty() {
        return;
    }

    let response = match serde_json::from_str::<GenerateRequest>(trimmed) {
        Err(e) => GenerateResponse::Error {
            message: format!("Invalid request JSON: {e}"),
        },
        Ok(request) => {
            let (reply_tx, reply_rx) = mpsc::channel();
            if work_tx.send(WorkItem { request, reply_tx }).is_err() {
                GenerateResponse::Error {
                    message: "Server shutting down".to_string(),
                }
            } else {
                reply_rx.recv().unwrap_or(GenerateResponse::Error {
                    message: "No response from server loop".to_string(),
                })
            }
        }
    };

    match serde_json::to_string(&response) {
        Err(e) => eprintln!("[unix] Serialize error: {e}"),
        Ok(json) => {
            let _ = write_half.write_all(json.as_bytes());
            let _ = write_half.write_all(b"\n");
        }
    }
}
