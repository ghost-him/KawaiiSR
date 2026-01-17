use crate::pipeline::task_meta::{ImageMeta, TaskType};
use crate::pipeline::tensor_batcher::BatcherInfo;
use crate::pipeline::tiling_utils::TilingInfo;
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashSet;
use ndarray::{s, Array3};
use std::sync::Arc;

/// 分块器接收的信息，包含处理后的图片数据和元数据
pub struct TilerInfo {
    pub task_id: usize,
    // 处理流ID（如果原图有多个通道需要分别处理，会生成多个处理流）
    // 例如：原图有Alpha，则会有流0(RGB)和流1(Alpha)
    pub task_type: TaskType,
    // 处理后的图片数据（CHW格式）
    pub image_data: Array3<f32>,
    // 图片元数据
    pub image_meta: Arc<ImageMeta>,
}

pub struct ImageTiler {
    _handle: std::thread::JoinHandle<()>,
}

impl ImageTiler {
    pub fn new(
        preprocessor_rx: Receiver<TilerInfo>,
        batcher_tx: Sender<BatcherInfo>,
        cancelled_tasks: Arc<DashSet<usize>>,
    ) -> Self {
        let mut inner = ImageTilerInner {
            preprocessor_rx,
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
    preprocessor_rx: Receiver<TilerInfo>,
    batcher_tx: Sender<BatcherInfo>,
    cancelled_tasks: Arc<DashSet<usize>>,
}

impl ImageTilerInner {
    fn execute(&mut self) {
        tracing::info!("[ImageTiler] Started");

        // 从预处理器接收图片数据进行切片
        while let Ok(preprocessor_output) = self.preprocessor_rx.recv() {
            let task_id = preprocessor_output.task_id;
            let image_data = preprocessor_output.image_data;
            let image_meta = preprocessor_output.image_meta;

            // 检查确认任务是否已被取消
            if self.cancelled_tasks.contains(&task_id) {
                tracing::info!("[ImageTiler] Task {} is cancelled", task_id);
                continue;
            }

            tracing::info!(
                "[ImageTiler] Processing task {} with shape: {:?}",
                task_id,
                image_data.shape()
            );

            let (c, h, w) = image_data.dim();
            let info = TilingInfo::new_with_config(
                h,
                w,
                image_meta.tile_height,
                image_meta.tile_width,
                image_meta.border,
                image_meta.overlap,
            );

            // 1. 创建大的填充后的数组，默认值为 0 (满足 0 填充要求)
            // 该数组尺寸是 tile 大小的整数倍
            let mut padded_data = Array3::zeros((c, info.padded_h, info.padded_w));

            // 2. 填充 reflect 数据到 padded_data 中。
            // 镜像范围是 original 图像及其外围 border 像素。
            for y in 0..(h + 2 * info.border) {
                for x in 0..(w + 2 * info.border) {
                    let src_y = reflect_index((y as isize) - (info.border as isize), h);
                    let src_x = reflect_index((x as isize) - (info.border as isize), w);

                    for channel in 0..c {
                        padded_data[[channel, y, x]] = image_data[[channel, src_y, src_x]];
                    }
                }
            }

            // 3. 将其按照 tile_h * tile_w 的方式完成切分，传送给 batcher
            let total_tiles = info.total_tiles();

            for i in 0..total_tiles {
                let (start_y, start_x) = info.get_tile_start(i);

                // 提取指定大小的块
                let tile_data = padded_data
                    .slice(s![
                        ..,
                        start_y..start_y + info.tile_h,
                        start_x..start_x + info.tile_w
                    ])
                    .to_owned();

                let batcher_info = BatcherInfo {
                    task_id,
                    tile_index: i,
                    task_type: preprocessor_output.task_type.clone(),
                    tile_data,
                    image_meta: image_meta.clone(),
                };

                if let Err(e) = self.batcher_tx.send(batcher_info) {
                    tracing::error!(
                        "[ImageTiler] Failed to send tile to batcher for task {}: {}",
                        task_id,
                        e
                    );
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
