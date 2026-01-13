use ndarray::Array4;
use std::sync::Arc;
use crate::{pipeline::image_meta::ImageMeta, sr_manager::ImageInfo};
use crossbeam_channel::{Receiver, Sender};
use tauri::async_runtime::Mutex;

pub struct StitcherInfo {
    pub task_id: Vec<usize>,
    pub tile_index: Vec<usize>,
    pub stitched_data: Array4<f32>,
    pub image_meta: Vec<Arc<ImageMeta>>,
}



#[derive(Debug)]
pub struct ImageStitcher {
    inner: Arc<Mutex<ImageStitcherInner>>,
}

impl ImageStitcher {
    pub fn new(onnx_rx: Receiver<StitcherInfo>, manager_tx: Sender<ImageInfo>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ImageStitcherInner::new(onnx_rx, manager_tx))),
        }
    }
}

#[derive(Debug)]
pub struct ImageStitcherInner {
    pub onnx_rx: Receiver<StitcherInfo>,
    pub manager_tx: Sender<ImageInfo>,
}

impl ImageStitcherInner {
    pub fn new(onnx_rx: Receiver<StitcherInfo>, manager_tx: Sender<ImageInfo>) -> Self {
        Self {
            onnx_rx,
            manager_tx,
        }
    }
}