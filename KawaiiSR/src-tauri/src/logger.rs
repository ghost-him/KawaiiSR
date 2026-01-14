use chrono::{Duration, Local};
use std::fs;
use std::panic;
use std::path::{Path, PathBuf};
use tracing::{error, info};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

pub fn init_logging(app_dir: PathBuf) {
    let log_dir = app_dir.join("logs");
    if !log_dir.exists() {
        fs::create_dir_all(&log_dir).unwrap_or_else(|e| {
            eprintln!("Failed to create log directory: {}", e);
        });
    }

    // Clean up old logs (older than 5 days)
    clean_old_logs(&log_dir, 5);

    let file_appender = RollingFileAppender::new(Rotation::DAILY, &log_dir, "kawaiisr.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    // Leak the guard to keep it alive for the duration of the program
    Box::leak(Box::new(_guard));

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort=warn,onnxruntime=warn"));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(std::io::stdout))
        .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
        .init();

    setup_panic_hook();

    info!("Logging initialized. Logs are stored in: {:?}", log_dir);
}

fn clean_old_logs(log_dir: &Path, keep_days: i64) {
    let now = Local::now();
    let threshold = now - Duration::days(keep_days);

    if let Ok(entries) = fs::read_dir(log_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        let modified_chrono: chrono::DateTime<Local> = modified.into();
                        if modified_chrono < threshold {
                            let _ = fs::remove_file(path);
                        }
                    }
                }
            }
        }
    }
}

fn setup_panic_hook() {
    panic::set_hook(Box::new(|panic_info| {
        let location = panic_info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown".to_string());
        let payload = panic_info.payload();
        let message = if let Some(s) = payload.downcast_ref::<&str>() {
            *s
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.as_str()
        } else {
            "no message"
        };

        error!("PANIC at {}: {}", location, message);

        // Also print to stderr for good measure
        eprintln!("PANIC at {}: {}", location, message);
    }));
}
