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
use crossbeam_channel::{bounded, unbounded, Sender};
use dashmap::{DashMap, DashSet};
use ndarray::Array4;
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

    pub async fn cancel_task(&self, task_id: usize) {
        let inner = self.inner.lock().await;
        inner.cancelled_tasks.insert(task_id);
        tracing::info!("[SRManager] Task {} marked as cancelled", task_id);
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
    // 被取消的任务集
    pub cancelled_tasks: Arc<DashSet<usize>>,
}

impl Default for SRManagerInner {
    fn default() -> Self {
        // 创建各 pipeline 之间的通道

        // Manager -> Tiler 的通道 (限制最多同时读取一个图片等待处理)
        // 改为较大的有界通道，因为现在发送的只是路径，不占内存
        let (manager_tx_tiler, manager_rx_tiler) = bounded(100);

        // Tiler -> Batcher 的通道 (限制 16 个切片在队列中)
        let (tiler_tx_batcher, tiler_rx_batcher) = bounded(16);

        // Batcher -> OnnxSession 的通道 (限制 2 个 Batch)
        let (batcher_tx_onnx, batcher_rx_onnx) = bounded(2);

        // OnnxSession -> Stitcher 的通道 (限制 16 个切片等待合并)
        let (onnx_tx_stitcher, onnx_rx_stitcher) = bounded(16);

        // Stitcher -> Manager 的通道
        let (stitcher_tx_manager, stitcher_rx_manager) = unbounded();

        // 创建各组件
        let results = Arc::new(DashMap::new());
        let cancelled_tasks = Arc::new(DashSet::new());

        let image_tiler = ImageTiler::new(manager_rx_tiler, tiler_tx_batcher, cancelled_tasks.clone());
        let tensor_batcher = TensorBatcher::new(tiler_rx_batcher, batcher_tx_onnx, cancelled_tasks.clone());
        let onnx_session = OnnxSession::new(batcher_rx_onnx, onnx_tx_stitcher, cancelled_tasks.clone());
        let image_stitcher = ImageStitcher::new(onnx_rx_stitcher, stitcher_tx_manager, cancelled_tasks.clone());

        // 监控最终结果
        let result_collector = ResultCollector::new(stitcher_rx_manager, results.clone(), cancelled_tasks.clone(), None);

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
            cancelled_tasks,
        }
    }
}

impl SRManagerInner {
    // 开始超分辨率一个图片
    pub fn run_inference(&mut self, sr_info: SRInfo) -> Result<usize> {
        // 1. 仅获取图片尺寸，不加载完整图片数据
        let (width, height) = image::image_dimensions(&sr_info.input_path)?;

        let current_task_id = self.id_generator.generate_id();

        // 计算并存储元数据
        let info = crate::pipeline::tiling_utils::TilingInfo::new(height as usize, width as usize);
        self.task_metadata.insert(
            current_task_id,
            TaskMetadata {
                total_tiles: info.total_tiles(),
            },
        );

        // 2. 封装信息并发送给 image_tiler，仅传递路径
        let tiler_info = TilerInfo {
            task_id: current_task_id,
            input_path: sr_info.input_path.clone(),
            scale_factor: sr_info.scale_factor,
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
