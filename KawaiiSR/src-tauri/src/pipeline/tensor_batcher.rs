use crate::pipeline::onnx_session::OnnxSessionInfo;
use crate::pipeline::task_meta::{ImageMeta, TaskType};
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashSet;
use ndarray::Array3;
use std::cmp::min;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

// 用于向TensorBatcher提供切片信息
// 单个切片表示为 Array3 (CHW)
pub struct BatcherInfo {
    pub task_id: usize,
    pub tile_index: usize,
    pub task_type: TaskType,
    pub tile_data: Array3<f32>,
    pub image_meta: Arc<ImageMeta>,
}

struct QueuedTile {
    task_id: usize,
    tile_index: usize,
    task_type: TaskType,
    tile_data: Array3<f32>,
    image_meta: Arc<ImageMeta>,
}

pub struct TensorBatcher {
    _handle: std::thread::JoinHandle<()>,
}

impl TensorBatcher {
    pub fn new(
        tiler_rx: Receiver<BatcherInfo>,
        onnx_tx: Sender<OnnxSessionInfo>,
        cancelled_tasks: Arc<DashSet<usize>>,
    ) -> Self {
        let mut inner = TensorBatcherInner {
            tiler_rx,
            onnx_tx,
            cancelled_tasks,
            queues: HashMap::new(),
            queue_timers: HashMap::new(),
        };

        let handle = std::thread::spawn(move || {
            inner.execute();
        });

        Self { _handle: handle }
    }
}

struct TensorBatcherInner {
    tiler_rx: Receiver<BatcherInfo>,
    onnx_tx: Sender<OnnxSessionInfo>,
    cancelled_tasks: Arc<DashSet<usize>>,
    queues: HashMap<String, VecDeque<QueuedTile>>,
    queue_timers: HashMap<String, Instant>,
}

impl TensorBatcherInner {
    // 获取当前批处理大小，后续可以根据性能动态调整
    fn get_batch_size(&self) -> usize {
        2
    }

    fn get_max_wait_time(&self) -> Duration {
        Duration::from_millis(50)
    }

    fn execute(&mut self) {
        tracing::info!("[TensorBatcher] Started");
        let batch_size = self.get_batch_size();
        let max_wait_time = self.get_max_wait_time();
        let recv_wait_time = Duration::from_millis(20);

        loop {
            match self.tiler_rx.recv_timeout(recv_wait_time) {
                Ok(info) => {
                    let model_name = info.image_meta.model_name.clone();
                    let queue = self
                        .queues
                        .entry(model_name.clone())
                        .or_insert_with(VecDeque::new);

                    // 如果是新的数据，则重置计时器（防止第一个batch一直无法运行）
                    if queue.is_empty() {
                        self.queue_timers.insert(model_name.clone(), Instant::now());
                    }
                    queue.push_back(QueuedTile {
                        task_id: info.task_id,
                        tile_index: info.tile_index,
                        task_type: info.task_type,
                        tile_data: info.tile_data,
                        image_meta: info.image_meta,
                    });
                }
                Err(RecvTimeoutError::Disconnected) => {
                    tracing::info!("[TensorBatcher] Channel disconnected, exiting");
                    break; // 退出循环
                }
                Err(_) => {}
            }
            // 2. 检查是否有新数据到达以重置计时器

            let model_names: Vec<String> = self.queues.keys().cloned().collect();
            for model_name in model_names {
                let queue = self.queues.get_mut(&model_name).unwrap();

                // 移除已取消任务
                queue.retain(|tile| !self.cancelled_tasks.contains(&tile.task_id));

                let up_to_batch = queue.len() >= batch_size;

                let timeout = if let Some(time) = self.queue_timers.get(&model_name) {
                    time.elapsed() >= max_wait_time
                } else {
                    false
                };

                // 只有要么达到批处理大小，要么超时且有数据时，才发送批处理
                if up_to_batch || (timeout && !queue.is_empty()) {
                    let current_batch_size = min(batch_size, queue.len());
                    let mut batch_task_ids = Vec::with_capacity(current_batch_size);
                    let mut batch_tile_indices = Vec::with_capacity(current_batch_size);
                    let mut batch_task_types = Vec::with_capacity(current_batch_size);
                    let mut batch_tiles_data = Vec::with_capacity(current_batch_size);
                    let mut batch_image_metas = Vec::with_capacity(current_batch_size);

                    for _ in 0..current_batch_size {
                        if let Some(tile) = queue.pop_front() {
                            batch_task_ids.push(tile.task_id);
                            batch_tile_indices.push(tile.tile_index);
                            batch_task_types.push(tile.task_type);
                            batch_tiles_data.push(tile.tile_data);
                            batch_image_metas.push(tile.image_meta);
                        }
                    }
                    // 重置计时器
                    self.queue_timers.insert(model_name.clone(), Instant::now());

                    if !batch_tiles_data.is_empty() {
                        // 堆叠张量
                        let views: Vec<_> = batch_tiles_data.iter().map(|t| t.view()).collect();
                        let batch_data = match ndarray::stack(ndarray::Axis(0), &views) {
                            Ok(data) => data,
                            Err(e) => {
                                tracing::error!(
                                    "[TensorBatcher] Failed to stack tiles for {}: {}",
                                    model_name,
                                    e
                                );
                                continue;
                            }
                        };

                        let onnx_info = OnnxSessionInfo {
                            task_id: batch_task_ids,
                            tile_index: batch_tile_indices,
                            task_type: batch_task_types,
                            batch_data,
                            image_meta: batch_image_metas,
                            model_name: model_name.clone(), // 确保传递模型名称
                        };

                        if let Err(e) = self.onnx_tx.send(onnx_info) {
                            tracing::error!(
                                "[TensorBatcher] Failed to send to ONNX session: {}",
                                e
                            );
                            return;
                        }
                    }
                }
            }
        }
        tracing::info!("[TensorBatcher] Stopped");
    }
}
