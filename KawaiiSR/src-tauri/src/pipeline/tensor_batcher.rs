use ndarray::Array3;
use crate::pipeline::image_meta::ImageMeta;
use std::sync::Arc;
use crossbeam_channel::{Receiver, Sender};
use tauri::async_runtime::Mutex;
use crate::pipeline::onnx_session::OnnxSessionInfo;


// 用于向TensorBatcher提供切片信息
// 单个切片表示为 Array3 (CHW)
pub struct BatcherInfo {
    pub task_id: usize,
    pub tile_index: usize,
    pub tile_data: Array3<f32>,
    pub image_meta: Arc<ImageMeta>,
}


#[derive(Debug)]
pub struct TensorBatcher {
    #[allow(dead_code)]
    inner: Arc<Mutex<TensorBatcherInner>>,
}

impl TensorBatcher {
    pub fn new(tiler_rx: Receiver<BatcherInfo>, onnx_tx: Sender<OnnxSessionInfo>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TensorBatcherInner::new(tiler_rx, onnx_tx))),
        }
    }
}

#[derive(Debug)]
pub struct TensorBatcherInner {
    pub tiler_rx: Receiver<BatcherInfo>,
    pub onnx_tx: Sender<OnnxSessionInfo>,
}

impl TensorBatcherInner {
    pub fn new(tiler_rx: Receiver<BatcherInfo>, onnx_tx: Sender<OnnxSessionInfo>) -> Self {
        Self {
            tiler_rx,
            onnx_tx,
        }
    }
}