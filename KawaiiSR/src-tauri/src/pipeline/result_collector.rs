use std::sync::Arc;
use tauri::async_runtime::Mutex;
use tauri::AppHandle;
use crate::sr_manager::ImageInfo;
use crossbeam_channel::Receiver;
use dashmap::DashMap;
use tauri::Emitter;

pub struct ResultCollector {
    _handle: std::thread::JoinHandle<()>,
    app_handle_store: Arc<Mutex<Option<AppHandle>>>,
}

impl ResultCollector {
    pub fn new(
        result_rx: Receiver<ImageInfo>,
        results: Arc<DashMap<usize, ImageInfo>>,
        app_handle: Option<AppHandle>,
    ) -> Self {
        let app_handle_store = Arc::new(Mutex::new(app_handle));
        let app_handle_store_inner = app_handle_store.clone();

        let mut inner = ResultCollectorInner {
            result_rx,
            results,
            app_handle: app_handle_store_inner,
        };

        let handle = std::thread::spawn(move || {
            inner.execute();
        });

        Self { _handle: handle, app_handle_store }
    }

    pub async fn set_app_handle(&self, handle: AppHandle) {
        let mut store = self.app_handle_store.lock().await;
        *store = Some(handle);
    }
}

struct ResultCollectorInner {
    result_rx: Receiver<ImageInfo>,
    results: Arc<DashMap<usize, ImageInfo>>,
    app_handle: Arc<Mutex<Option<AppHandle>>>,
}

impl ResultCollectorInner {
    fn execute(&mut self) {
        tracing::info!("[ResultCollector] Started");

        while let Ok(image_info) = self.result_rx.recv() {
            let task_id = image_info.task_id;
            tracing::info!(
                "[ResultCollector] Received result for task {} with shape: {:?}",
                task_id,
                image_info.image_data.shape()
            );

            // 0. 调试: 保存到本地硬盘
            {
                let data = &image_info.image_data;
                let shape = data.shape();
                let height = shape[2];
                let width = shape[3];
                
                let mut pixels = Vec::with_capacity(height * width * 3);
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..3 {
                            let val = (data[[0, c, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
                            pixels.push(val);
                        }
                    }
                }
                
                if let Some(img) = image::RgbImage::from_raw(width as u32, height as u32, pixels) {
                    let filename = format!("output_{}.png", task_id);
                    if let Err(e) = img.save(&filename) {
                        tracing::error!("[ResultCollector] Failed to save debug image {}: {}", filename, e);
                    } else {
                        tracing::info!("[ResultCollector] Debug image saved to {}", filename);
                    }
                }
            }

            // 1. 保存到内部存储
            self.results.insert(task_id, image_info);

            // 2. 交互外部模块: 通知前端任务完成
            let app_handle_clone = self.app_handle.clone();
            tauri::async_runtime::block_on(async move {
                let handle_lock = app_handle_clone.lock().await;
                if let Some(handle) = &*handle_lock {
                    if let Err(e) = handle.emit("sr-task-completed", task_id) {
                        tracing::error!("[ResultCollector] Failed to emit event: {}", e);
                    }
                }
            });
        }

        tracing::info!("[ResultCollector] Stopped");
    }
}
