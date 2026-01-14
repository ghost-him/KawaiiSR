use ndarray::{s, Array3};
use crossbeam_channel::{Sender, Receiver};
use std::sync::Arc;
use crate::pipeline::image_meta::ImageMeta;
use crate::pipeline::tensor_batcher::BatcherInfo;
use crate::pipeline::tiling_utils::{BORDER, TILE_SIZE, TilingInfo};

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
        tracing::info!("[ImageTiler] Started");
        
        // 从管道接收图片数据进行切片
        while let Ok(tiler_info) = self.manager_rx.recv() {
            tracing::info!(
                "[ImageTiler] Processing task {} with image shape: {:?}",
                tiler_info.task_id,
                tiler_info.image_data.shape()
            );
            
            let (c, h, w) = tiler_info.image_data.dim();
            let info = TilingInfo::new(h, w);
            
            // 1. 创建大的填充后的数组，默认值为 0 (满足 0 填充要求)
            // 该数组尺寸是 TILE_SIZE 的整数倍
            let mut padded_data = Array3::zeros((c, info.padded_h, info.padded_w));
            
            // 2. 填充 reflect 数据到 padded_data 中。
            // 镜像范围是 original 图像及其外围 BORDER (64) 像素。
            for y in 0..(h + 2 * BORDER) {
                for x in 0..(w + 2 * BORDER) {
                    let src_y = reflect_index((y as isize) - (BORDER as isize), h);
                    let src_x = reflect_index((x as isize) - (BORDER as isize), w);
                    
                    for channel in 0..c {
                        padded_data[[channel, y, x]] = tiler_info.image_data[[channel, src_y, src_x]];
                    }
                }
            }

            // 3. 将其按照 128*128 的方式完成切分，传送给 batcher
            let total_tiles = info.total_tiles();
            for i in 0..total_tiles {
                let (start_y, start_x) = info.get_tile_start(i);
                
                // 提取 128x128 的块
                let tile_data = padded_data.slice(s![.., start_y..start_y + TILE_SIZE, start_x..start_x + TILE_SIZE]).to_owned();

                let batcher_info = BatcherInfo {
                    task_id: tiler_info.task_id,
                    tile_index: i,
                    tile_data,
                    image_meta: tiler_info.image_meta.clone(),
                };
                
                if let Err(e) = self.batcher_tx.send(batcher_info) {
                    tracing::error!("[ImageTiler] Failed to send to batcher: {}", e);
                    break;
                }
            }
        }
        
        tracing::info!("[ImageTiler] Stopped");
    }
}

/// 镜像索引计算，处理任意长度的 padding
fn reflect_index(mut i: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let last = (len - 1) as isize;
    loop {
        if i < 0 {
            i = -i;
        } else if i > last {
            i = last - (i - last);
        } else {
            return i as usize;
        }
    }
}
