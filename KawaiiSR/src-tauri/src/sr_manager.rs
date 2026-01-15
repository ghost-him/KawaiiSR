use crate::id_generator::IDGenerator;
use crate::pipeline::image_meta::ImageMeta;
use crate::pipeline::result_collector::ResultCollector;
use crate::pipeline::{
    image_stitcher::ImageStitcher,
    image_tiler::{ImageTiler, TilerInfo},
    onnx_session::OnnxSession,
    tensor_batcher::TensorBatcher,
};
use anyhow::Result;
use crossbeam_channel::{unbounded, Sender};
use dashmap::DashMap;
use image::GenericImageView;
use ndarray::{Array3, Array4};
use std::sync::Arc;
use tauri::{async_runtime::Mutex, AppHandle};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ModelName {
    #[default]
    KawaiiSR,
}

/// 用于对外表示的信息，
/// 可见范围：外部调用模块，SRManager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SRInfo {
    // 输入图像路径
    pub input_path: String,
    // 模型名称
    pub model_name: ModelName,
    // 放大的倍数
    pub scale_factor: u32,
    // 保存路径（如果为空，则表示保存到内存中，等待获取）
    pub output_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    pub total_tiles: usize,
}

/// 对内表示的信息，可见范围：SRManager 以及 image_stitcher
/// 表示image_stticher工作结束，传回给SRManager的信息
#[derive(Clone)]
pub struct ImageInfo {
    // 该图片任务的唯一标识符
    pub task_id: usize,
    // 图片的内容 (NCHW 格式)
    pub image_data: Array4<f32>,
    // 图片的元数据
    pub image_meta: Arc<ImageMeta>,
}

pub struct SRManager {
    // 用于管理超分辨率管道的结构体
    inner: Arc<Mutex<SRManagerInner>>,
}

impl Default for SRManager {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SRManagerInner::default())),
        }
    }
}

impl SRManager {
    pub async fn run_inference(&self, sr_info: SRInfo) -> Result<usize> {
        let mut inner = self.inner.lock().await;
        inner.run_inference(sr_info)
    }

    // 获取该图片的结果
    pub async fn get_result(&self, task_id: usize) -> Option<ImageInfo> {
        let inner = self.inner.lock().await;
        inner.get_result(task_id).await
    }

    // 获取任务元数据
    pub async fn get_task_metadata(&self, task_id: usize) -> Option<TaskMetadata> {
        let inner = self.inner.lock().await;
        inner.task_metadata.get(&task_id).map(|v| v.clone())
    }

    // 设置 AppHandle 以便发射事件
    pub async fn set_app_handle(&self, handle: AppHandle) {
        let inner = self.inner.lock().await;
        inner.result_collector.set_app_handle(handle.clone()).await;
        inner.image_stitcher.set_app_handle(handle).await;
    }
}

pub struct SRManagerInner {
    // 将图片分割
    pub image_tiler: ImageTiler,
    // 将图片合并成一个batch来运行
    pub tensor_batcher: TensorBatcher,
    // Onnx推理
    pub onnx_session: OnnxSession,
    // 图片恢复
    pub image_stitcher: ImageStitcher,
    // 任务标识符生成器
    pub id_generator: Arc<IDGenerator>,
    // 向 Tiler 发送数据的通道 (manager 只维护这个)
    pub tiler_tx: Sender<TilerInfo>,
    // 任务结果存储 (Key: task_id)
    pub results: Arc<DashMap<usize, ImageInfo>>,
    // 任务元数据存储 (Key: task_id)
    pub task_metadata: DashMap<usize, TaskMetadata>,
    // 结果收集器 (后台线程监控)
    pub result_collector: ResultCollector,
}

impl Default for SRManagerInner {
    fn default() -> Self {
        // 创建各 pipeline 之间的通道

        // Manager -> Tiler 的通道
        let (manager_tx_tiler, manager_rx_tiler) = unbounded();

        // Tiler -> Batcher 的通道
        let (tiler_tx_batcher, tiler_rx_batcher) = unbounded();

        // Batcher -> OnnxSession 的通道
        let (batcher_tx_onnx, batcher_rx_onnx) = unbounded();

        // OnnxSession -> Stitcher 的通道
        let (onnx_tx_stitcher, onnx_rx_stitcher) = unbounded();

        // Stitcher -> Manager 的通道
        let (stitcher_tx_manager, stitcher_rx_manager) = unbounded();

        // 创建各组件
        let results = Arc::new(DashMap::new());

        let image_tiler = ImageTiler::new(manager_rx_tiler, tiler_tx_batcher);
        let tensor_batcher = TensorBatcher::new(tiler_rx_batcher, batcher_tx_onnx);
        let onnx_session = OnnxSession::new(batcher_rx_onnx, onnx_tx_stitcher);
        let image_stitcher = ImageStitcher::new(onnx_rx_stitcher, stitcher_tx_manager);

        // 监控最终结果
        let result_collector = ResultCollector::new(stitcher_rx_manager, results.clone(), None);

        Self {
            image_tiler,
            tensor_batcher,
            onnx_session,
            image_stitcher,
            id_generator: Arc::new(IDGenerator::default()),
            tiler_tx: manager_tx_tiler,
            results,
            task_metadata: DashMap::new(),
            result_collector,
        }
    }
}

impl SRManagerInner {
    // 开始超分辨率一个图片
    pub fn run_inference(&mut self, sr_info: SRInfo) -> Result<usize> {
        // 1. 读取当前的图片
        let img = image::open(&sr_info.input_path)?;
        let (width, height) = img.dimensions();
        let rgb_img = img.to_rgb8();
        let image_meta = Arc::new(ImageMeta {
            original_width: width as usize,
            original_height: height as usize,
            scale_factor: sr_info.scale_factor,
        });

        // 2. 将其变成ndarray [3, H, W] f32 归一化 (CHW)
        let mut array = Array3::<f32>::zeros((3, height as usize, width as usize));
        for (x, y, pixel) in rgb_img.enumerate_pixels() {
            for c in 0..3 {
                array[[c, y as usize, x as usize]] = pixel[c] as f32 / 255.0;
            }
        }

        let current_task_id = self.id_generator.generate_id();

        // 计算并存储元数据
        let info = crate::pipeline::tiling_utils::TilingInfo::new(height as usize, width as usize);
        self.task_metadata.insert(
            current_task_id,
            TaskMetadata {
                total_tiles: info.total_tiles(),
            },
        );

        // 3. 封装信息并发送给 image_tiler
        let tiler_info = TilerInfo {
            task_id: current_task_id,
            image_data: array,
            image_meta: image_meta,
        };
        self.tiler_tx
            .send(tiler_info)
            .map_err(|e| anyhow::anyhow!("Failed to send to tiler: {}", e))?;

        Ok(current_task_id)
    }

    // 获取该图片的结果
    pub async fn get_result(&self, task_id: usize) -> Option<ImageInfo> {
        self.results
            .get(&task_id)
            .map(|entry| entry.value().clone())
    }

    // 删除图片的数据
    pub async fn remove_result(&self, task_id: usize) {
        self.results.remove(&task_id);
    }
}
