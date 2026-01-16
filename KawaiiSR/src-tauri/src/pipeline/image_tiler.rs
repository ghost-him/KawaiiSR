use crate::pipeline::image_meta::ImageMeta;
use crate::pipeline::tensor_batcher::BatcherInfo;
use crate::pipeline::tiling_utils::{TilingInfo, BORDER, TILE_SIZE};
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashSet;
use image::GenericImageView;
use ndarray::{s, Array3};
use std::sync::Arc;

// 用于向ImageTiler提供切片信息
pub struct TilerInfo {
    // 该图片任务的唯一标识符
    pub task_id: usize,
    // 图片输入路径
    pub input_path: String,
    // 模型缩放倍数
    pub scale_factor: u32,
}

pub struct ImageTiler {
    _handle: std::thread::JoinHandle<()>,
}

impl ImageTiler {
    pub fn new(
        manager_rx: Receiver<TilerInfo>,
        batcher_tx: Sender<BatcherInfo>,
        cancelled_tasks: Arc<DashSet<usize>>,
    ) -> Self {
        let mut inner = ImageTilerInner {
            manager_rx,
            batcher_tx,
            cancelled_tasks,
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
    cancelled_tasks: Arc<DashSet<usize>>,
}

impl ImageTilerInner {
    fn execute(&mut self) {
        tracing::info!("[ImageTiler] Started");

        // 从管道接收图片数据进行切片
        while let Ok(tiler_info) = self.manager_rx.recv() {
            // 检查确认任务是否已被取消
            if self.cancelled_tasks.contains(&tiler_info.task_id) {
                tracing::info!("[ImageTiler] Task {} is cancelled, skipping", tiler_info.task_id);
                continue;
            }

            tracing::info!(
                "[ImageTiler] Processing task {} with image: {}",
                tiler_info.task_id,
                tiler_info.input_path
            );

            // 1. 在此处加载并预处理图片
            let (image_data, image_meta) = match self.load_image(&tiler_info) {
                Ok(data) => data,
                Err(e) => {
                    tracing::error!(
                        "[ImageTiler] Failed to load image {}: {}",
                        tiler_info.input_path,
                        e
                    );
                    continue;
                }
            };

            let (c, h, w) = image_data.dim();
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
                        padded_data[[channel, y, x]] = image_data[[channel, src_y, src_x]];
                    }
                }
            }

            // 3. 将其按照 128*128 的方式完成切分，传送给 batcher
            let total_tiles = info.total_tiles();

            for i in 0..total_tiles {
                let (start_y, start_x) = info.get_tile_start(i);

                // 提取 128x128 的块
                let tile_data = padded_data
                    .slice(s![
                        ..,
                        start_y..start_y + TILE_SIZE,
                        start_x..start_x + TILE_SIZE
                    ])
                    .to_owned();

                let batcher_info = BatcherInfo {
                    task_id: tiler_info.task_id,
                    tile_index: i,
                    tile_data: tile_data,
                    image_meta: image_meta.clone(),
                };

                if let Err(e) = self.batcher_tx.send(batcher_info) {
                    tracing::error!(
                        "[ImageTiler] Failed to send tile to batcher for task {}: {}",
                        tiler_info.task_id,
                        e
                    );
                    break;
                }
            }
        }

        tracing::info!("[ImageTiler] Stopped");
    }

    fn load_image(&self, info: &TilerInfo) -> anyhow::Result<(Array3<f32>, Arc<ImageMeta>)> {
        let img = image::open(&info.input_path)?;
        let (width, height) = img.dimensions();
        let rgb_img = img.to_rgb8();
        let image_meta = Arc::new(ImageMeta {
            original_width: width as usize,
            original_height: height as usize,
            scale_factor: info.scale_factor,
        });

        // 归一化 (CHW)
        let mut array = Array3::<f32>::zeros((3, height as usize, width as usize));
        for (x, y, pixel) in rgb_img.enumerate_pixels() {
            for c in 0..3 {
                array[[c, y as usize, x as usize]] = pixel[c] as f32 / 255.0;
            }
        }

        Ok((array, image_meta))
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
