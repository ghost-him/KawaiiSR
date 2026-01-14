use ndarray::Array3;
use crossbeam_channel::{Sender, Receiver};
use std::sync::Arc;
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

pub struct ImageTiler {
    _handle: std::thread::JoinHandle<()>,
}

impl ImageTiler {
    pub fn new(manager_rx: Receiver<TilerInfo>, batcher_tx: Sender<BatcherInfo>) -> Self {
        let mut inner = ImageTilerInner {
            manager_rx,
            batcher_tx,
        };

        let handle = std::thread::spawn(move || {
            inner.execute();
        });

        Self { _handle: handle }
    }
}

struct ImageTilerInner {
    manager_rx: Receiver<TilerInfo>,
    batcher_tx: Sender<BatcherInfo>,
}

impl ImageTilerInner {
    fn execute(&mut self) {
        println!("[ImageTiler] Started");
        
        // 从管道接收图片数据进行切片
        while let Ok(tiler_info) = self.manager_rx.recv() {
            println!(
                "[ImageTiler] Processing task {} with image shape: {:?}",
                tiler_info.task_id,
                tiler_info.image_data.shape()
            );
            
            // 当前简化版本：不进行切片，直接将整张图片作为一个 tile 传递
            // TODO: 未来可以实现真正的切片逻辑（tile_size、overlap等）
            let tile_data = tiler_info.image_data;
            
            let batcher_info = BatcherInfo {
                task_id: tiler_info.task_id,
                tile_index: 0, // 只有一个 tile
                tile_data,
                image_meta: tiler_info.image_meta.clone(),
            };
            
            if let Err(e) = self.batcher_tx.send(batcher_info) {
                eprintln!("[ImageTiler] Failed to send to batcher: {}", e);
                break;
            }
        }
        
        println!("[ImageTiler] Stopped");
    }
}