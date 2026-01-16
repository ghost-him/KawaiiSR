use crate::pipeline::tiling_utils::{TilingInfo, BORDER, OVERLAP, TILE_SIZE};
use crate::{pipeline::image_meta::ImageMeta, sr_manager::ImageInfo};
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashSet;
use ndarray::{s, Array3, Array4, Axis};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tauri::async_runtime::Mutex;
use tauri::{AppHandle, Emitter};

pub struct StitcherInfo {
    pub task_id: Vec<usize>,
    pub tile_index: Vec<usize>,
    pub stitched_data: Array4<f32>,
    pub image_meta: Vec<Arc<ImageMeta>>,
}

#[derive(Serialize, Clone)]
pub struct ProgressPayload {
    pub task_id: usize,
    pub completed_tiles: usize,
    pub total_tiles: usize,
}

pub struct ImageStitcher {
    _handle: std::thread::JoinHandle<()>,
    app_handle_store: Arc<Mutex<Option<AppHandle>>>,
}

impl ImageStitcher {
    pub fn new(
        onnx_rx: Receiver<StitcherInfo>,
        manager_tx: Sender<ImageInfo>,
        cancelled_tasks: Arc<DashSet<usize>>,
    ) -> Self {
        let app_handle_store = Arc::new(Mutex::new(None));
        let app_handle_store_inner = app_handle_store.clone();

        let mut inner = ImageStitcherInner {
            onnx_rx,
            manager_tx,
            cancelled_tasks,
            tasks: HashMap::new(),
            app_handle: app_handle_store_inner,
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

struct TaskProgress {
    accumulated_data: Array3<f32>,
    received_count: usize,
    info: TilingInfo,
}

struct ImageStitcherInner {
    onnx_rx: Receiver<StitcherInfo>,
    manager_tx: Sender<ImageInfo>,
    cancelled_tasks: Arc<DashSet<usize>>,
    tasks: HashMap<usize, TaskProgress>,
    app_handle: Arc<Mutex<Option<AppHandle>>>,
}

impl ImageStitcherInner {
    fn execute(&mut self) {
        tracing::info!("[ImageStitcher] Started");
        while let Ok(stitcher_info) = self.onnx_rx.recv() {
            let batch_size = stitcher_info.stitched_data.shape()[0];

            // 第一步：清理当前批次中所有已取消任务的中间状态
            // 这样可以避免被取消的任务永久堆积在 self.tasks 中
            for i in 0..batch_size {
                let task_id = stitcher_info.task_id[i];
                if self.cancelled_tasks.contains(&task_id) {
                    if self.tasks.remove(&task_id).is_some() {
                        tracing::info!(
                            "[ImageStitcher] Task {} cancelled, cleared context",
                            task_id
                        );
                    }
                }
            }

            // 第二步：记录需要处理的任务索引（跳过已取消的任务）
            let mut valid_indices: Vec<usize> = Vec::new();

            for i in 0..batch_size {
                if !self.cancelled_tasks.contains(&stitcher_info.task_id[i]) {
                    valid_indices.push(i);
                }
            }

            // 如果批次中所有任务都已取消，直接跳过
            if valid_indices.is_empty() {
                continue;
            }

            // 第三步：处理有效任务
            for &i in &valid_indices {
                let task_id = stitcher_info.task_id[i];
                let tile_index = stitcher_info.tile_index[i];
                let image_meta = &stitcher_info.image_meta[i];
                let scale = image_meta.scale_factor as usize;

                // 获取或创建任务进度
                let progress = self.tasks.entry(task_id).or_insert_with(|| {
                    let info =
                        TilingInfo::new(image_meta.original_height, image_meta.original_width);
                    let shape = (3, info.padded_h * scale, info.padded_w * scale);

                    TaskProgress {
                        accumulated_data: Array3::zeros(shape),
                        received_count: 0,
                        info,
                    }
                });

                // 提取当前 tile 的数据
                let tile_data = stitcher_info.stitched_data.index_axis(Axis(0), i);
                let (start_y, start_x) = progress.info.get_tile_start(tile_index);

                // 计算裁剪大小（只保留中心区域，去除重叠部分）
                let out_tile_size = TILE_SIZE * scale;
                let out_overlap = OVERLAP * scale;
                let crop = out_overlap / 2;
                let effective_size = out_tile_size - 2 * crop;

                let sy = start_y * scale;
                let sx = start_x * scale;

                // 将 tile 的中心 96*96 (或对应缩放后的尺寸) 区域直接存入结果
                let cropped_tile = tile_data.slice(s![
                    ..,
                    crop..out_tile_size - crop,
                    crop..out_tile_size - crop
                ]);
                let mut acc_slice = progress.accumulated_data.slice_mut(s![
                    ..,
                    sy + crop..sy + crop + effective_size,
                    sx + crop..sx + crop + effective_size
                ]);
                acc_slice.assign(&cropped_tile);

                progress.received_count += 1;

                // 发送进度更新
                let current_progress = progress.received_count;
                let total_tiles = progress.info.total_tiles();
                let app_handle_clone = self.app_handle.clone();

                tauri::async_runtime::block_on(async move {
                    let handle_lock = app_handle_clone.lock().await;
                    if let Some(handle) = &*handle_lock {
                        let payload = ProgressPayload {
                            task_id,
                            completed_tiles: current_progress,
                            total_tiles,
                        };
                        if let Err(e) = handle.emit("sr-task-progress", payload) {
                            tracing::error!("[ImageStitcher] Failed to emit progress: {}", e);
                        }
                    }
                });

                // 检查是否完成
                if progress.received_count == progress.info.total_tiles() {
                    let final_progress = self.tasks.remove(&task_id).unwrap();

                    // 裁剪掉填充和边缘
                    let h_start = BORDER * scale;
                    let h_end = (BORDER + image_meta.original_height) * scale;
                    let w_start = BORDER * scale;
                    let w_end = (BORDER + image_meta.original_width) * scale;

                    let final_image = final_progress
                        .accumulated_data
                        .slice(s![.., h_start..h_end, w_start..w_end])
                        .to_owned();

                    // 封装并发送给 manager
                    let image_info = ImageInfo {
                        task_id,
                        image_data: final_image.insert_axis(Axis(0)),
                        image_meta: image_meta.clone(),
                    };

                    if let Err(e) = self.manager_tx.send(image_info) {
                        tracing::error!(
                            "[ImageStitcher] Failed to send result to manager for task {}: {}",
                            task_id,
                            e
                        );
                    } else {
                        tracing::info!(
                            "[ImageStitcher] Task {} completed and sent to collector",
                            task_id
                        );
                    }
                }
            }
        }

        tracing::info!("[ImageStitcher] Stopped");
    }
}
