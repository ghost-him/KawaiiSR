use ndarray::Array3;
use crossbeam_channel::{Sender, Receiver};
use std::sync::Arc;
use tauri::async_runtime::Mutex;
use crate::pipeline::image_meta::ImageMeta;
use crate::pipeline::tensor_batcher::BatcherInfo;

// 用于向ImageTiler提供切片信息
pub struct TilerInfo {
    // 该图片任务的唯一标识符
    pub task_id: usize,
    // 图片的内容 (CHW 格式，单个图片)
    pub image_data: Array3<f32>,
    // 图片的元数据
    pub image_meta: Arc<ImageMeta>,
}


#[derive(Debug)]
pub struct ImageTiler {
    #[allow(dead_code)]
    inner : Arc<Mutex<ImageTilerInner>>,
}

impl ImageTiler {
    pub fn new(manager_rx: Receiver<TilerInfo>, batcher_tx: Sender<BatcherInfo>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ImageTilerInner::new(manager_rx, batcher_tx))),
        }
    }
}

#[derive(Debug)]
pub struct ImageTilerInner {
    pub manager_rx: Receiver<TilerInfo>,
    pub batcher_tx: Sender<BatcherInfo>,
}

impl ImageTilerInner {
    pub fn new(manager_rx: Receiver<TilerInfo>, batcher_tx: Sender<BatcherInfo>) -> Self {
        Self {
            manager_rx,
            batcher_tx,
        }
    }
}