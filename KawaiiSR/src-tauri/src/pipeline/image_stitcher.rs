use ndarray::{Array3, Array4, Axis, s};
use std::sync::Arc;
use std::collections::HashMap;
use crate::{pipeline::image_meta::ImageMeta, sr_manager::ImageInfo};
use crossbeam_channel::{Receiver, Sender};
use crate::pipeline::tiling_utils::{BORDER, TILE_SIZE, TilingInfo, OVERLAP};

pub struct StitcherInfo {
    pub task_id: Vec<usize>,
    pub tile_index: Vec<usize>,
    pub stitched_data: Array4<f32>,
    pub image_meta: Vec<Arc<ImageMeta>>,
}

pub struct ImageStitcher {
    _handle: std::thread::JoinHandle<()>,
}

impl ImageStitcher {
    pub fn new(onnx_rx: Receiver<StitcherInfo>, manager_tx: Sender<ImageInfo>) -> Self {
        let mut inner = ImageStitcherInner {
            onnx_rx,
            manager_tx,
            tasks: HashMap::new(),
        };

        let handle = std::thread::spawn(move || {
            inner.execute();
        });

        Self { _handle: handle }
    }
}

struct TaskProgress {
    accumulated_data: Array3<f32>,
    received_count: usize,
    info: TilingInfo,
}

struct ImageStitcherInner {
    onnx_rx: Receiver<StitcherInfo>,
    manager_tx: Sender<ImageInfo>,
    tasks: HashMap<usize, TaskProgress>,
}

impl ImageStitcherInner {
    fn execute(&mut self) {
        tracing::info!("[ImageStitcher] Started");
        while let Ok(stitcher_info) = self.onnx_rx.recv() {
            let batch_size = stitcher_info.stitched_data.shape()[0];
            
            for i in 0..batch_size {
                let task_id = stitcher_info.task_id[i];
                let tile_index = stitcher_info.tile_index[i];
                let image_meta = &stitcher_info.image_meta[i];
                let scale = image_meta.scale_factor as usize;
                
                // 获取或创建任务进度
                let progress = self.tasks.entry(task_id).or_insert_with(|| {
                    let info = TilingInfo::new(image_meta.original_height, image_meta.original_width);
                    let shape = (3, info.padded_h * scale, info.padded_w * scale);

                    TaskProgress {
                        accumulated_data: Array3::zeros(shape),
                        received_count: 0,
                        info,
                    }
                });
                
                // 提取当前 tile 的数据
                let tile_data = stitcher_info.stitched_data.index_axis(Axis(0), i);
                let (start_y, start_x) = progress.info.get_tile_start(tile_index);
                
                // 计算裁剪大小（只保留中心区域，去除重叠部分）
                let out_tile_size = TILE_SIZE * scale;
                let out_overlap = OVERLAP * scale;
                let crop = out_overlap / 2;
                let effective_size = out_tile_size - 2 * crop;
                
                let sy = start_y * scale;
                let sx = start_x * scale;
                
                // 将 tile 的中心 96*96 (或对应缩放后的尺寸) 区域直接存入结果
                let cropped = tile_data.slice(s![.., crop..out_tile_size - crop, crop..out_tile_size - crop]);
                let mut acc_slice = progress.accumulated_data.slice_mut(s![.., sy + crop .. sy + crop + effective_size, sx + crop .. sx + crop + effective_size]);
                acc_slice.assign(&cropped);
                
                progress.received_count += 1;
                
                // 检查是否完成
                if progress.received_count == progress.info.total_tiles() {
                    let progress = self.tasks.remove(&task_id).unwrap();
                    
                    // 裁剪掉填充和边缘
                    let h_start = BORDER * scale;
                    let h_end = (BORDER + image_meta.original_height) * scale;
                    let w_start = BORDER * scale;
                    let w_end = (BORDER + image_meta.original_width) * scale;
                    
                    let cropped = progress.accumulated_data.slice(s![.., h_start..h_end, w_start..w_end]).to_owned();
                    
                    // 封装并发送给 manager
                    let image_info = ImageInfo {
                        task_id,
                        image_data: cropped.insert_axis(Axis(0)),
                        image_meta: image_meta.clone(),
                    };
                    
                    if let Err(e) = self.manager_tx.send(image_info) {
                        tracing::error!("[ImageStitcher] Failed to send result to manager: {}", e);
                    }
                }
            }
        }
        
        tracing::info!("[ImageStitcher] Stopped");
    }
}
