pub mod app_state;
pub mod config;
pub mod id_generator;
pub mod logger;
pub mod pipeline;
pub mod sr_manager;

use crate::app_state::AppState;
use crate::sr_manager::{SRInfo, TaskMetaStruct};
use std::sync::Arc;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn get_task_metadata(
    state: tauri::State<'_, Arc<AppState>>,
    task_id: usize,
) -> Result<TaskMetaStruct, String> {
    let manager = state.sr_pipline.clone();
    manager
        .get_task_metadata(task_id)
        .await
        .ok_or_else(|| format!("Metadata for task {} not found", task_id))
}

#[tauri::command]
async fn run_super_resolution(
    state: tauri::State<'_, Arc<AppState>>,
    input_path: String,
    model_name: String,
    output_path: Option<String>,
    overlap: Option<usize>,
    border: Option<usize>,
) -> Result<usize, String> {
    tracing::info!(
        "Command 'run_super_resolution' invoked for {}, model: {}, output: {:?}, overlap: {:?}, border: {:?}",
        input_path,
        model_name,
        output_path,
        overlap,
        border
    );

    let manager = state.sr_pipline.clone();

    let sr_info = SRInfo {
        input_path,
        model_name,
        overlap,
        border,
        output_path,
    };

    match manager.run_inference(sr_info).await {
        Ok(task_id) => {
            tracing::info!("✓ Super-resolution task {} started successfully", task_id);
            Ok(task_id)
        }
        Err(e) => {
            let err_msg = format!("SR Error: {:?}", e);
            tracing::error!("{}", err_msg);
            Err(err_msg)
        }
    }
}

#[tauri::command]
async fn cancel_super_resolution(
    state: tauri::State<'_, Arc<AppState>>,
    task_id: usize,
) -> Result<(), String> {
    tracing::info!(
        "Command 'cancel_super_resolution' invoked for task {}",
        task_id
    );
    let manager = state.sr_pipline.clone();
    manager.cancel_task(task_id).await;
    Ok(())
}

#[tauri::command]
async fn get_image_data(path: String) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_result_image(
    state: tauri::State<'_, Arc<AppState>>,
    task_id: usize,
) -> Result<Vec<u8>, String> {
    let manager = state.sr_pipline.clone();

    if let Some(info) = manager.get_result(task_id).await {
        // Convert to PNG bytes
        let data = &info.image_data;
        let shape = data.shape();
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let mut pixels = Vec::with_capacity(height * width * channels);
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    pixels.push(data[[0, c, y, x]] as u8);
                }
            }
        }

        let img_dynamic = if channels == 4 {
            image::RgbaImage::from_raw(width as u32, height as u32, pixels)
                .map(image::DynamicImage::ImageRgba8)
        } else {
            image::RgbImage::from_raw(width as u32, height as u32, pixels)
                .map(image::DynamicImage::ImageRgb8)
        };

        if let Some(img) = img_dynamic {
            let mut cursor = std::io::Cursor::new(Vec::new());
            img.write_to(&mut cursor, image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            Ok(cursor.into_inner())
        } else {
            Err("Failed to create image from raw pixels".to_string())
        }
    } else {
        Err("Result not found or not yet ready".to_string())
    }
}

#[tauri::command]
async fn save_result_image(
    state: tauri::State<'_, Arc<AppState>>,
    task_id: usize,
    output_path: String,
) -> Result<(), String> {
    state.sr_pipline.save_image(task_id, output_path).await
}

use tauri::Manager;

/// 获取可用的模型列表
#[tauri::command]
async fn get_available_models(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<Vec<String>, String> {
    let manager = state.sr_pipline.clone();
    let models = manager.get_available_models().await;
    Ok(models)
}

/// 获取默认模型名称
#[tauri::command]
async fn get_default_model(state: tauri::State<'_, Arc<AppState>>) -> Result<String, String> {
    let manager = state.sr_pipline.clone();
    Ok(manager.get_default_model().await)
}

/// 获取特定模型的配置
#[tauri::command]
async fn get_model_config(
    state: tauri::State<'_, Arc<AppState>>,
    model_name: String,
) -> Result<crate::config::model_config::ModelConfig, String> {
    let manager = state.sr_pipline.clone();
    manager
        .get_model_config(model_name.clone())
        .await
        .ok_or_else(|| format!("Model config for {} not found", model_name))
}

#[tauri::command]
async fn set_auto_save_dir(
    _state: tauri::State<'_, Arc<AppState>>,
    _path: Option<String>,
) -> Result<(), String> {
    Err("set_auto_save_dir is deprecated, use output_path in run_super_resolution".to_string())
}

#[tauri::command]
async fn get_auto_save_dir(
    _state: tauri::State<'_, Arc<AppState>>,
) -> Result<Option<String>, String> {
    Ok(None)
}

#[tauri::command]
async fn list_images_in_folder(path: String) -> Result<Vec<String>, String> {
    let mut images = Vec::new();
    let dir = std::fs::read_dir(path).map_err(|e| e.to_string())?;

    for entry in dir {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "webp" {
                    images.push(path.to_string_lossy().to_string());
                }
            }
        }
    }
    // Sort images for consistent processing order
    images.sort();
    Ok(images)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize logging
    let log_dir = std::env::var("APPDATA")
        .map(|s| std::path::PathBuf::from(s).join("KawaiiSR"))
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    logger::init_logging(log_dir);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            run_super_resolution,
            cancel_super_resolution,
            get_image_data,
            get_result_image,
            save_result_image,
            get_task_metadata,
            get_available_models,
            get_default_model,
            get_model_config,
            set_auto_save_dir,
            get_auto_save_dir,
            list_images_in_folder,
        ])
        .manage(Arc::new(AppState::default()))
        .setup(|app| {
            // 获取 AppState
            let state = app.state::<Arc<AppState>>();

            // 将 AppHandle 注入到 SRManager 的 ResultCollector 中
            let manager = state.sr_pipline.clone();
            let handle = app.handle().clone();
            let manager_clone = manager.clone();

            // 在异步运行时中执行注入
            tauri::async_runtime::spawn(async move {
                manager_clone.set_app_handle(handle).await;
                tracing::info!("[Setup] AppHandle injected into SRManager");
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
