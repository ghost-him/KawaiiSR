use crate::config::model_config::NormalizationRange;
use crate::config::ConfigManager;
use crate::pipeline::image_tiler::TilerInfo;
use crate::pipeline::task_meta::TaskType;
use crate::pipeline::task_meta::{ImageMeta, ImageType};
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashSet;
use image::GenericImageView;
use ndarray::Array3;
use std::sync::Arc;
/// 预处理器输入的信息
pub struct PreprocessorInfo {
    pub task_id: usize,
    pub input_path: String,
    pub scale_factor: u32,
    pub model_name: String,
    pub tile_width: usize,
    pub tile_height: usize,
    pub border: usize,
    pub overlap: usize,
}

pub struct ImagePreprocessor {
    _handle: std::thread::JoinHandle<()>,
}

impl ImagePreprocessor {
    pub fn new(
        manager_rx: Receiver<PreprocessorInfo>,
        tiler_tx: Sender<TilerInfo>,
        cancelled_tasks: Arc<DashSet<usize>>,
        config_manager: Arc<ConfigManager>,
    ) -> Self {
        let mut inner = ImagePreprocessorInner {
            manager_rx,
            tiler_tx,
            cancelled_tasks,
            config_manager,
        };

        let handle = std::thread::spawn(move || {
            inner.execute();
        });

        Self { _handle: handle }
    }
}

struct ImagePreprocessorInner {
    manager_rx: Receiver<PreprocessorInfo>,
    tiler_tx: Sender<TilerInfo>,
    cancelled_tasks: Arc<DashSet<usize>>,
    config_manager: Arc<ConfigManager>,
}

impl ImagePreprocessorInner {
    fn execute(&mut self) {
        tracing::info!("[ImagePreprocessor] Started");

        while let Ok(preproc_info) = self.manager_rx.recv() {
            // 检查任务是否已取消
            if self.cancelled_tasks.contains(&preproc_info.task_id) {
                tracing::info!(
                    "[ImagePreprocessor] Task {} is cancelled, skipping",
                    preproc_info.task_id
                );
                continue;
            }

            tracing::info!(
                "[ImagePreprocessor] Processing task {} with image: {}",
                preproc_info.task_id,
                preproc_info.input_path
            );

            // 加载图片
            match self.load_and_preprocess_image(&preproc_info) {
                Ok(outputs) => {
                    for output in outputs {
                        if let Err(e) = self.tiler_tx.send(output) {
                            tracing::error!(
                                "[ImagePreprocessor] Failed to send preprocessed image to tiler: {}",
                                e
                            );
                            break;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        "[ImagePreprocessor] Failed to preprocess image {}: {}",
                        preproc_info.input_path,
                        e
                    );
                }
            }
        }

        tracing::info!("[ImagePreprocessor] Stopped");
    }

    fn load_and_preprocess_image(&self, info: &PreprocessorInfo) -> anyhow::Result<Vec<TilerInfo>> {
        let img = image::open(&info.input_path)?;
        let (width, height) = img.dimensions();

        // 获取模型配置以应用归一化
        let model_config = self.config_manager.get_model(&info.model_name)?;

        let mut outputs = vec![];
        // 如果是rgba图片，那么将该图片
        if img.has_alpha() {
            match img.to_rgba8() {
                rgba_img => {
                    // 提取
                    let mut rgb_data = Array3::<f32>::zeros((3, height as usize, width as usize));
                    let mut alpha_data = Array3::<f32>::zeros((3, height as usize, width as usize));
                    for (x, y, pixel) in rgba_img.enumerate_pixels() {
                        // 重复3次，得到一个3通道的，由输入图片alpha组成的图片
                        for c in 0..3 {
                            // 完成rgb的提取
                            rgb_data[[c, y as usize, x as usize]] = self
                                .apply_normalization(pixel[c] as f32, &model_config.normalization);
                            // 完成alpha的提取
                            alpha_data[[c, y as usize, x as usize]] = self
                                .apply_normalization(pixel[3] as f32, &model_config.normalization);
                        }
                    }

                    tracing::info!(
                        "[ImagePreprocessor] Task {} has alpha channel, generated 2 sub image",
                        info.task_id
                    );

                    let image_meta = Arc::new(ImageMeta {
                        original_width: width as usize,
                        original_height: height as usize,
                        scale_factor: model_config.scale,
                        model_name: info.model_name.clone(),
                        tile_width: info.tile_width,
                        tile_height: info.tile_height,
                        border: info.border,
                        overlap: info.overlap,
                    });

                    // 流 0: RGB 数据
                    outputs.push(TilerInfo {
                        task_id: info.task_id,
                        task_type: TaskType::Image(ImageType::RGBA(false)),
                        image_data: rgb_data,
                        image_meta: image_meta.clone(),
                    });

                    // 流 1: Alpha 数据
                    outputs.push(TilerInfo {
                        task_id: info.task_id,
                        task_type: TaskType::Image(ImageType::RGBA(true)),
                        image_data: alpha_data,
                        image_meta: image_meta.clone(),
                    });
                }
            }
        } else {
            // 如果只是普通的rgb图片
            match img.to_rgb8() {
                rgb_img => {
                    let mut rgb_data = Array3::<f32>::zeros((3, height as usize, width as usize));
                    for (x, y, pixel) in rgb_img.enumerate_pixels() {
                        for c in 0..3 {
                            rgb_data[[c, y as usize, x as usize]] = self
                                .apply_normalization(pixel[c] as f32, &model_config.normalization);
                        }
                    }

                    tracing::info!("[ImagePreprocessor] Task {} is RGB image", info.task_id);

                    let image_meta = Arc::new(ImageMeta {
                        original_width: width as usize,
                        original_height: height as usize,
                        scale_factor: model_config.scale,
                        model_name: info.model_name.clone(),
                        tile_width: info.tile_width,
                        tile_height: info.tile_height,
                        border: info.border,
                        overlap: info.overlap,
                    });

                    outputs.push(TilerInfo {
                        task_id: info.task_id,
                        task_type: TaskType::Image(ImageType::RGB),
                        image_data: rgb_data,
                        image_meta: image_meta.clone(),
                    });
                }
            }
        }

        Ok(outputs)
    }

    /// 应用归一化到像素数据
    fn apply_normalization(
        &self,
        pixel_value: f32,
        normalization: &crate::config::model_config::NormalizationConfig,
    ) -> f32 {
        let normalized = match normalization.range {
            NormalizationRange::ZeroToOne => pixel_value / 255.0,
            NormalizationRange::MinusOneToOne => (pixel_value / 255.0) * 2.0 - 1.0,
            NormalizationRange::ZeroTo255 => pixel_value,
        };

        if let (Some(mean), Some(std)) = (&normalization.mean, &normalization.std) {
            // 应用均值和标准差归一化
            (normalized - mean[0]) / std[0] // 假设RGB通道相同
        } else {
            normalized
        }
    }
}
