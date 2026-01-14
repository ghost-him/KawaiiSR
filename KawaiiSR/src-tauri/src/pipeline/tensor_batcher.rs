use ndarray::Array3;
use crate::pipeline::image_meta::ImageMeta;
use std::sync::Arc;
use crossbeam_channel::{Receiver, Sender};
use crate::pipeline::onnx_session::OnnxSessionInfo;

// 用于向TensorBatcher提供切片信息
// 单个切片表示为 Array3 (CHW)
pub struct BatcherInfo {
    pub task_id: usize,
    pub tile_index: usize,
    pub tile_data: Array3<f32>,
    pub image_meta: Arc<ImageMeta>,
}

pub struct TensorBatcher {
    _handle: std::thread::JoinHandle<()>,
}

impl TensorBatcher {
    pub fn new(tiler_rx: Receiver<BatcherInfo>, onnx_tx: Sender<OnnxSessionInfo>) -> Self {
        let mut inner = TensorBatcherInner {
            tiler_rx,
            onnx_tx,
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
}

impl TensorBatcherInner {
    fn execute(&mut self) {
        println!("[TensorBatcher] Started");
        
        // 从管道接收切片数据，进行批处理
        // 当前简化版本：每次只处理一个切片，不进行批合并
        // TODO: 未来可以实现真正的批处理（收集多个切片组成更大的batch）
        while let Ok(batcher_info) = self.tiler_rx.recv() {
            println!(
                "[TensorBatcher] Processing task {} tile {} with shape: {:?}",
                batcher_info.task_id,
                batcher_info.tile_index,
                batcher_info.tile_data.shape()
            );
            
            // 将 CHW 转换为 NCHW (batch size = 1)
            let batch_data = batcher_info.tile_data
                .insert_axis(ndarray::Axis(0));
            
            let onnx_info = OnnxSessionInfo {
                task_id: vec![batcher_info.task_id],
                tile_index: vec![batcher_info.tile_index],
                batch_data,
                image_meta: vec![batcher_info.image_meta],
            };
            
            if let Err(e) = self.onnx_tx.send(onnx_info) {
                eprintln!("[TensorBatcher] Failed to send to ONNX session: {}", e);
                break;
            }
        }
        
        println!("[TensorBatcher] Stopped");
    }
}