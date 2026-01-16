use crate::pipeline::image_meta::ImageMeta;
use crate::pipeline::onnx_session::OnnxSessionInfo;
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashSet;
use ndarray::Array3;
use std::cmp::min;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

// 用于向TensorBatcher提供切片信息
// 单个切片表示为 Array3 (CHW)
pub struct BatcherInfo {
    pub task_id: usize,
    pub tile_index: usize,
    pub tile_data: Array3<f32>,
    pub image_meta: Arc<ImageMeta>,
}

struct QueuedTile {
    task_id: usize,
    tile_index: usize,
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
            queue: VecDeque::new(),
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
    queue: VecDeque<QueuedTile>,
}

impl TensorBatcherInner {
    // 获取当前批处理大小，后续可以根据性能动态调整
    fn get_batch_size(&self) -> usize {
        2
    }

    fn execute(&mut self) {
        tracing::info!("[TensorBatcher] Started");
        let batch_size = self.get_batch_size();

        loop {
            // 1. 尽可能从接收端读取数据并放入队列
            while let Ok(batcher_info) = self.tiler_rx.try_recv() {
                self.queue.push_back(QueuedTile {
                    task_id: batcher_info.task_id,
                    tile_index: batcher_info.tile_index,
                    tile_data: batcher_info.tile_data,
                    image_meta: batcher_info.image_meta,
                });
                if self.queue.len() >= batch_size {
                    break;
                }
            }

            // 2. 如果队列中切片不足一个 batch，尝试短时间等待更多数据来填充 batch
            if self.queue.len() < batch_size {
                if self.queue.is_empty() {
                    // 队列完全为空，无限期阻塞直到第一个切片到来
                    match self.tiler_rx.recv() {
                        Ok(batcher_info) => {
                            self.queue.push_back(QueuedTile {
                                task_id: batcher_info.task_id,
                                tile_index: batcher_info.tile_index,
                                tile_data: batcher_info.tile_data,
                                image_meta: batcher_info.image_meta,
                            });
                        }
                        Err(_) => break, // 通道关闭
                    }
                }

                // 此时队列已经至少有一个切片，尝试在短时间内继续等待更多切片以填满 batch
                while self.queue.len() < batch_size {
                    // 使用 50ms 等待超时。如果 50ms 内没有新切片，则直接进入下一步处理已有切片。
                    match self.tiler_rx.recv_timeout(Duration::from_millis(50)) {
                        Ok(batcher_info) => {
                            self.queue.push_back(QueuedTile {
                                task_id: batcher_info.task_id,
                                tile_index: batcher_info.tile_index,
                                tile_data: batcher_info.tile_data,
                                image_meta: batcher_info.image_meta,
                            });
                        }
                        Err(_) => break, // 等待超时或通道关闭
                    }
                }
            }

            // 3. 准备处理前，过滤掉队列中已被取消任务的切片（处理前统一检查）
            self.queue
                .retain(|tile| !self.cancelled_tasks.contains(&tile.task_id));

            if self.queue.is_empty() {
                continue;
            }

            // 3. 从队列中取出指定数量的切片进行批处理
            let current_batch_size = min(batch_size, self.queue.len());
            let mut batch_task_ids = Vec::with_capacity(current_batch_size);
            let mut batch_tile_indices = Vec::with_capacity(current_batch_size);
            let mut batch_tiles_data = Vec::with_capacity(current_batch_size);
            let mut batch_image_metas = Vec::with_capacity(current_batch_size);

            for _ in 0..current_batch_size {
                if let Some(tile) = self.queue.pop_front() {
                    batch_task_ids.push(tile.task_id);
                    batch_tile_indices.push(tile.tile_index);
                    batch_tiles_data.push(tile.tile_data);
                    batch_image_metas.push(tile.image_meta);
                } else {
                    break;
                }
            }

            if batch_tiles_data.is_empty() {
                continue;
            }

            // 4. 将所有 CHW (Array3) 堆叠成 NCHW (Array4)
            let views: Vec<_> = batch_tiles_data.iter().map(|t| t.view()).collect();
            let batch_data = match ndarray::stack(ndarray::Axis(0), &views) {
                Ok(data) => data,
                Err(e) => {
                    tracing::error!("[TensorBatcher] Failed to stack tiles: {}", e);
                    continue;
                }
            };

            let onnx_info = OnnxSessionInfo {
                task_id: batch_task_ids,
                tile_index: batch_tile_indices,
                batch_data,
                image_meta: batch_image_metas,
            };

            if let Err(e) = self.onnx_tx.send(onnx_info) {
                tracing::error!("[TensorBatcher] Failed to send to ONNX session: {}", e);
                break;
            }
        }

        tracing::info!("[TensorBatcher] Stopped");
    }
}
