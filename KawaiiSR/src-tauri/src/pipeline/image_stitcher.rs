use ndarray::{Array4, Axis};
use std::sync::Arc;
use crate::{pipeline::image_meta::ImageMeta, sr_manager::ImageInfo};
use crossbeam_channel::{Receiver, Sender};
use image::{ImageBuffer, Rgb};

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
        };

        let handle = std::thread::spawn(move || {
            inner.execute();
        });

        Self { _handle: handle }
    }
}

struct ImageStitcherInner {
    onnx_rx: Receiver<StitcherInfo>,
    manager_tx: Sender<ImageInfo>,
}

impl ImageStitcherInner {
    fn execute(&mut self) {
        println!("[ImageStitcher] Started");
        
        // 从管道接收 ONNX 输出结果，进行拼接并保存
        while let Ok(stitcher_info) = self.onnx_rx.recv() {
            println!(
                "[ImageStitcher] Processing batch with {} tiles, output shape: {:?}",
                stitcher_info.task_id.len(),
                stitcher_info.stitched_data.shape()
            );
            
            if let Err(e) = self.process_output(stitcher_info) {
                eprintln!("[ImageStitcher] Failed to process output: {:?}", e);
                // 继续处理下一批
            }
        }
        
        println!("[ImageStitcher] Stopped");
    }
    
    fn process_output(&mut self, stitcher_info: StitcherInfo) -> Result<(), Box<dyn std::error::Error>> {
        // 当前简化版本：假设只有一个 tile，直接转换为图像并发送给 manager
        // TODO: 未来实现多 tile 拼接逻辑
        
        let batch_size = stitcher_info.stitched_data.shape()[0];
        
        for i in 0..batch_size {
            let task_id = stitcher_info.task_id[i];
            let tile_index = stitcher_info.tile_index[i];
            
            println!(
                "[ImageStitcher] Processing task {} tile {}",
                task_id, tile_index
            );
            
            // 提取单个图像 [C, H, W]
            let single_output = stitcher_info.stitched_data.index_axis(Axis(0), i);
            let _out_c = single_output.shape()[0];
            let out_h = single_output.shape()[1];
            let out_w = single_output.shape()[2];
            
            // 保存为 PNG 文件（用于临时调试）
            let out_array = single_output
                .permuted_axes([1, 2, 0]) // CHW -> HWC
                .as_standard_layout()
                .to_owned()
                .mapv(|x| (x * 255.0).clamp(0.0, 255.0) as u8);
            
            let out_img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                out_array.into_raw_vec_and_offset().0,
            )
            .ok_or("Failed to create ImageBuffer from raw output data")?;
            
            let output_path = format!("output_task_{}_tile_{}.png", task_id, tile_index);
            out_img.save(&output_path)?;
            println!("[ImageStitcher] Saved to: {}", output_path);
            
            // 将数据转换为 NCHW 格式并发送给 manager
            let image_data = single_output.insert_axis(Axis(0)).to_owned();
            
            let image_info = ImageInfo {
                task_id,
                image_data,
                image_meta: stitcher_info.image_meta[i].clone(),
            };
            
            if let Err(e) = self.manager_tx.send(image_info) {
                eprintln!("[ImageStitcher] Failed to send to manager: {}", e);
            }
        }
        
        Ok(())
    }
}