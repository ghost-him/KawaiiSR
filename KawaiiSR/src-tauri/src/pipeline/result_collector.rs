use crate::config::ConfigManager;
use crate::pipeline::task_meta::{ImageType, TaskType};
use crate::sr_manager::ImageInfo;
use crossbeam_channel::Receiver;
use dashmap::{DashMap, DashSet};
use ndarray::{stack, Axis};
use std::path::Path;
use std::sync::Arc;
use tauri::async_runtime::Mutex;
use tauri::AppHandle;
use tauri::Emitter;

pub fn save_image_to_file(info: &ImageInfo, output_path: &Path) -> Result<(), String> {
    let data = &info.image_data;
    let shape = data.shape();
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    let mut pixels = Vec::with_capacity(height * width * channels);
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                // 数据已经预先反归一化到 [0, 255]
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
        // 确保目录存在
        if let Some(parent) = output_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return Err(format!("Failed to create directory: {}", e));
            }
        }
        img.save(output_path).map_err(|e| e.to_string())
    } else {
        Err("Failed to create image from raw pixels".to_string())
    }
}

pub struct ResultCollector {
    _handle: std::thread::JoinHandle<()>,
    app_handle_store: Arc<Mutex<Option<AppHandle>>>,
}

struct PartialTask {
    rgb: Option<ImageInfo>,
    alpha: Option<ImageInfo>,
}

impl ResultCollector {
    pub fn new(
        result_rx: Receiver<ImageInfo>,
        results: Arc<DashMap<usize, ImageInfo>>,
        cancelled_tasks: Arc<DashSet<usize>>,
        app_handle: Option<AppHandle>,
        config_manager: Arc<ConfigManager>,
    ) -> Self {
        let app_handle_store = Arc::new(Mutex::new(app_handle));
        let app_handle_store_inner = app_handle_store.clone();

        let mut inner = ResultCollectorInner {
            result_rx,
            results,
            partial_results: std::collections::HashMap::new(),
            cancelled_tasks,
            app_handle: app_handle_store_inner,
            config_manager,
        };

        let handle = std::thread::spawn(move || {
            inner.execute();
        });

        Self {
            _handle: handle,
            app_handle_store,
        }
    }

    pub async fn set_app_handle(&self, handle: AppHandle) {
        let mut store = self.app_handle_store.lock().await;
        *store = Some(handle);
    }
}

struct ResultCollectorInner {
    result_rx: Receiver<ImageInfo>,
    results: Arc<DashMap<usize, ImageInfo>>,
    partial_results: std::collections::HashMap<usize, PartialTask>,
    cancelled_tasks: Arc<DashSet<usize>>,
    app_handle: Arc<Mutex<Option<AppHandle>>>,
    config_manager: Arc<ConfigManager>,
}

impl ResultCollectorInner {
    fn execute(&mut self) {
        tracing::info!("[ResultCollector] Started");

        while let Ok(image_info) = self.result_rx.recv() {
            let task_id = image_info.task_id;

            // 检查确认任务是否已取消
            if self.cancelled_tasks.contains(&task_id) {
                tracing::info!(
                    "[ResultCollector] Dropping result for cancelled task {}",
                    task_id
                );
                self.partial_results.remove(&task_id);
                continue;
            }

            tracing::info!(
                "[ResultCollector] Received result for task {} ({:?}) with shape: {:?}",
                task_id,
                image_info.task_type,
                image_info.image_data.shape()
            );

            // 处理合并逻辑
            let final_result = match &image_info.task_type {
                TaskType::Image(ImageType::RGB) => {
                    // 普通 RGB 图片，直接完成
                    Some(image_info)
                }
                TaskType::Image(ImageType::RGBA(is_alpha)) => {
                    let entry = self.partial_results.entry(task_id).or_insert(PartialTask {
                        rgb: None,
                        alpha: None,
                    });

                    if *is_alpha {
                        entry.alpha = Some(image_info);
                    } else {
                        entry.rgb = Some(image_info);
                    }

                    // 检查是否都到了
                    if entry.rgb.is_some() && entry.alpha.is_some() {
                        let partial = self.partial_results.remove(&task_id).unwrap();
                        let rgb_info = partial.rgb.unwrap();
                        let alpha_info = partial.alpha.unwrap();

                        // 合并 RGB 和 Alpha
                        let rgb_data = rgb_info.image_data.index_axis(Axis(0), 0);
                        let alpha_data = alpha_info.image_data.index_axis(Axis(0), 0);

                        let mut combined_channels = Vec::new();
                        // RGB
                        for c in 0..3 {
                            combined_channels.push(rgb_data.index_axis(Axis(0), c).to_owned());
                        }
                        // Alpha (取第一个通道)
                        combined_channels.push(alpha_data.index_axis(Axis(0), 0).to_owned());

                        let views: Vec<_> = combined_channels.iter().map(|ch| ch.view()).collect();
                        let merged_data = stack(Axis(0), &views)
                            .expect("Failed to stack channels in ResultCollector")
                            .insert_axis(Axis(0));

                        Some(ImageInfo {
                            task_id,
                            task_type: TaskType::Image(ImageType::RGBA(false)), // 最终结果
                            image_data: merged_data,
                            image_meta: rgb_info.image_meta.clone(),
                        })
                    } else {
                        None
                    }
                }
                TaskType::Video(frame_idx) => {
                    // 视频帧处理逻辑可以在这里扩展
                    // 目前直接存入
                    tracing::info!(
                        "[ResultCollector] Received frame {} for task {}",
                        frame_idx,
                        task_id
                    );
                    Some(image_info)
                }
            };

            if let Some(mut final_info) = final_result {
                // 获取模型配置以执行反归一化
                if let Ok(model_config) = self
                    .config_manager
                    .get_model(&final_info.image_meta.model_name)
                {
                    tracing::info!(
                        "[ResultCollector] Applying denormalization for task {}",
                        task_id
                    );
                    // 在存储前直接对数据进行反归一化处理
                    final_info
                        .image_data
                        .mapv_inplace(|v| model_config.denormalize(v));
                }

                // 1. 保存到内部存储 (现在存储的是 [0, 255] 范围的数据)
                self.results.insert(task_id, final_info.clone());

                // 2. 自动保存逻辑 (如果设置了 output_path)
                if let Some(output_path) = &final_info.image_meta.output_path {
                    tracing::info!(
                        "[ResultCollector] Auto-saving task {} to {}",
                        task_id,
                        output_path
                    );

                    if let Err(e) = save_image_to_file(&final_info, Path::new(output_path)) {
                        tracing::error!("[ResultCollector] Auto-save failed: {}", e);
                    }
                }

                // 3. 交互外部模块: 通知前端任务完成
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
        }

        tracing::info!("[ResultCollector] Stopped");
    }
}
