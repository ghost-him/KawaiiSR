pub mod sr_manager;
pub mod app_state;
pub mod pipeline;
pub mod id_generator;

use std::sync::Arc;
use crate::app_state::AppState;
use crate::sr_manager::{SRInfo, ModelName};

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn run_super_resolution(state: tauri::State<'_, Arc<AppState>>) -> Result<String, String> {
    println!("Command 'run_super_resolution' invoked");
    
    let manager = state.sr_pipline.clone();

    // 默认使用项目根目录下的 test.png 进行测试
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let mut img_path = cwd.join("../test.png");
    if !img_path.exists() {
        img_path = cwd.join("test.png");
    }

    if !img_path.exists() {
        return Err(format!("Input image not found at {:?}. CWD is {:?}", img_path, cwd));
    }

    let sr_info = SRInfo {
        input_path: img_path.to_string_lossy().to_string(),
        model_name: ModelName::KawaiiSR,
        scale_factor: 2, // 默认放大倍数
        output_path: None,
    };

    match manager.run_inference(sr_info).await {
        Ok(task_id) => {
            println!("✓ Super-resolution task {} started successfully", task_id);
            Ok(format!("Super-resolution task started. Task ID: {}", task_id))
        },
        Err(e) => {
            let err_msg = format!("SR Error: {:?}", e);
            eprintln!("{}", err_msg);
            Err(err_msg)
        }
    }
}

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, run_super_resolution])
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
                    println!("[Setup] AppHandle injected into SRManager");
                });
                
            
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
