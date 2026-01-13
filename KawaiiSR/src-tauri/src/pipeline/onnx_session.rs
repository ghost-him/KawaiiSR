use ndarray::Array4;
use std::sync::Arc;
use crate::pipeline::image_meta::ImageMeta;
use crossbeam_channel::{Receiver, Sender};
use tauri::async_runtime::Mutex;
use crate::pipeline::image_stitcher::StitcherInfo;

pub struct OnnxSessionInfo {
    // 可能会将不同任务的多个切片一起送入ONNX进行推理，所以要区分不同的batch上是哪一个任务的切片
    pub task_id: Vec<usize>,
    pub tile_index: Vec<usize>,
    // 进行推理的切片数据 (NCHW 格式)
    pub batch_data: Array4<f32>,
    // 每个切片对应的图片元数据
    pub image_meta: Vec<Arc<ImageMeta>>,
}


#[derive(Debug)]
pub struct OnnxSession {
    #[allow(dead_code)]
    inner: Arc<Mutex<OnnxSessionInner>>,
}

impl OnnxSession {
    pub fn new(batcher_rx: Receiver<OnnxSessionInfo>, stitcher_tx: Sender<StitcherInfo>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(OnnxSessionInner::new(batcher_rx, stitcher_tx))),
        }
    }
}

#[derive(Debug)]
pub struct OnnxSessionInner {
    pub batcher_rx: Receiver<OnnxSessionInfo>,
    pub stitcher_tx: Sender<StitcherInfo>,
}

impl OnnxSessionInner {
    pub fn new(batcher_rx: Receiver<OnnxSessionInfo>, stitcher_tx: Sender<StitcherInfo>) -> Self {
        Self {
            batcher_rx,
            stitcher_tx,
        }
    }
}