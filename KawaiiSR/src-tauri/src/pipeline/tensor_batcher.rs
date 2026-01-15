use crate::pipeline::image_meta::ImageMeta;
use crate::pipeline::onnx_session::OnnxSessionInfo;
use crossbeam_channel::{Receiver, Sender};
use ndarray::Array3;
use std::cmp::min;
use std::collections::VecDeque;
use std::sync::Arc;

// 用于向TensorBatcher提供切片信息
// 单个切片表示为 Array3 (CHW)
pub struct BatcherInfo {
    pub task_id: usize,
    pub tile_index: Vec<usize>,
    pub tile_data: Vec<Array3<f32>>,
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
    pub fn new(tiler_rx: Receiver<BatcherInfo>, onnx_tx: Sender<OnnxSessionInfo>) -> Self {
        let mut inner = TensorBatcherInner {
            tiler_rx,
            onnx_tx,
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
    queue: VecDeque<QueuedTile>,
}

impl TensorBatcherInner {
    // 获取当前批处理大小，后续可以根据性能动态调整
    fn get_batch_size(&self) -> usize {
        2
    }

    fn execute(&mut self) {
        tracing::info!("[TensorBatcher] Started");

        loop {
            // 1. 尽可能从接收端读取数据并放入队列
            // 使用 try_recv 避免阻塞，这样可以在没有新输入时也能处理队列中的数据
            while let Ok(batcher_info) = self.tiler_rx.try_recv() {
                let num_tiles = batcher_info.tile_data.len();
                tracing::info!(
                    "[TensorBatcher] Queuing task {} with {} tiles",
                    batcher_info.task_id,
                    num_tiles
                );

                for (idx, data) in batcher_info
                    .tile_index
                    .into_iter()
                    .zip(batcher_info.tile_data.into_iter())
                {
                    self.queue.push_back(QueuedTile {
                        task_id: batcher_info.task_id,
                        tile_index: idx,
                        tile_data: data,
                        image_meta: batcher_info.image_meta.clone(),
                    });
                }
            }

            // 2. 如果队列为空，则阻塞等待新数据
            if self.queue.is_empty() {
                match self.tiler_rx.recv() {
                    Ok(batcher_info) => {
                        let num_tiles = batcher_info.tile_data.len();
                        tracing::info!(
                            "[TensorBatcher] Received and queuing task {} with {} tiles",
                            batcher_info.task_id,
                            num_tiles
                        );

                        for (idx, data) in batcher_info
                            .tile_index
                            .into_iter()
                            .zip(batcher_info.tile_data.into_iter())
                        {
                            self.queue.push_back(QueuedTile {
                                task_id: batcher_info.task_id,
                                tile_index: idx,
                                tile_data: data,
                                image_meta: batcher_info.image_meta.clone(),
                            });
                        }
                    }
                    Err(_) => {
                        // 通道关闭，退出循环
                        break;
                    }
                }
            }

            // 3. 从队列中取出指定数量的切片进行批处理
            let current_batch_size = min(self.get_batch_size(), self.queue.len());
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
