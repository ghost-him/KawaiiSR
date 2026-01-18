use crate::config::ConfigManager;
use crate::id_generator::IDGenerator;
use crate::pipeline::result_collector::ResultCollector;
use crate::pipeline::task_meta::{ImageMeta, TaskType};
use crate::pipeline::{
    image_preprocessor::{ImagePreprocessor, PreprocessorInfo},
    image_stitcher::ImageStitcher,
    image_tiler::ImageTiler,
    model_manager::ModelManager,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SRInfo {
    // 输入图像路径
    pub input_path: String,
    // 模型名称
    pub model_name: String,
    // 放大的倍数
    pub scale_factor: u32,
    // 保存路径（如果为空，则表示保存到内存中，等待获取）
    pub output_path: Option<String>,
}

// 这个是用于向前端传数据的任务元数据结构体，不是内部使用的
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetaStruct {
    pub total_tiles: usize,
    pub input_size: u64,
    pub input_width: u32,
    pub input_height: u32,
}

/// 对内表示的信息，可见范围：SRManager 以及 image_stitcher
/// 表示image_stticher工作结束，传回给SRManager的信息
#[derive(Clone)]
pub struct ImageInfo {
    // 该图片任务的唯一标识符
    pub task_id: usize,
    // 该流的类型
    pub task_type: TaskType,
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
    pub async fn get_task_metadata(&self, task_id: usize) -> Option<TaskMetaStruct> {
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

    // 获取可用模型列表
    pub async fn get_available_models(&self) -> Vec<String> {
        let inner = self.inner.lock().await;
        inner.config_manager.list_models()
    }

    // 获取默认模型名称
    pub async fn get_default_model(&self) -> String {
        let inner = self.inner.lock().await;
        inner.config_manager.default_model_name().to_string()
    }
}

pub struct SRManagerInner {
    // 图片预处理器
    pub image_preprocessor: ImagePreprocessor,
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
    // 向 Preprocessor 发送数据的通道 (manager 只维护这个)
    pub preprocessor_tx: Sender<PreprocessorInfo>,
    // 任务结果存储 (Key: task_id)
    pub results: Arc<DashMap<usize, ImageInfo>>,
    // 任务元数据存储 (Key: task_id)
    pub task_metadata: DashMap<usize, TaskMetaStruct>,
    // 结果收集器 (后台线程监控)
    pub result_collector: ResultCollector,
    // 被取消的任务集
    pub cancelled_tasks: Arc<DashSet<usize>>,
    // 配置管理器：管理所有模型配置
    pub config_manager: Arc<ConfigManager>,
    // 模型管理器：缓存 ONNX 会话
    pub model_manager: Arc<ModelManager>,
}

impl Default for SRManagerInner {
    fn default() -> Self {
        // 加载模型配置
        let config_manager = match Self::load_config() {
            Ok(cm) => Arc::new(cm),
            Err(e) => {
                tracing::error!("[SRManager] Failed to load config: {}", e);
                // 降级处理：如果配置加载失败，使用空的配置管理器
                // 实际应用中可能需要更优雅的错误处理
                panic!("Failed to load model configuration: {}", e);
            }
        };

        // 创建各 pipeline 之间的通道

        // Manager -> Preprocessor 的通道
        let (manager_tx_preprocessor, manager_rx_preprocessor) = bounded(100);

        // Preprocessor -> Tiler 的通道 (预处理可能输出多个流，如RGB+Alpha)
        let (preprocessor_tx_tiler, preprocessor_rx_tiler) = bounded(32);

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
        let model_manager = Arc::new(ModelManager::new(config_manager.clone()));

        let image_preprocessor = ImagePreprocessor::new(
            manager_rx_preprocessor,
            preprocessor_tx_tiler,
            cancelled_tasks.clone(),
        );
        let image_tiler = ImageTiler::new(
            preprocessor_rx_tiler,
            tiler_tx_batcher,
            cancelled_tasks.clone(),
        );
        let tensor_batcher =
            TensorBatcher::new(tiler_rx_batcher, batcher_tx_onnx, cancelled_tasks.clone());
        let onnx_session = OnnxSession::new(
            batcher_rx_onnx,
            onnx_tx_stitcher,
            cancelled_tasks.clone(),
            model_manager.clone(),
        );
        let image_stitcher = ImageStitcher::new(
            onnx_rx_stitcher,
            stitcher_tx_manager,
            cancelled_tasks.clone(),
        );

        // 监控最终结果
        let result_collector = ResultCollector::new(
            stitcher_rx_manager,
            results.clone(),
            cancelled_tasks.clone(),
            None,
        );

        Self {
            image_preprocessor,
            image_tiler,
            tensor_batcher,
            onnx_session,
            image_stitcher,
            id_generator: Arc::new(IDGenerator::default()),
            preprocessor_tx: manager_tx_preprocessor,
            results,
            task_metadata: DashMap::new(),
            result_collector,
            cancelled_tasks,
            config_manager,
            model_manager,
        }
    }
}

impl SRManagerInner {
    /// 加载模型配置
    fn load_config() -> Result<ConfigManager> {
        // 尝试从多个位置加载配置文件
        let config_paths = vec!["models.toml", "src-tauri/models.toml"];

        for path in config_paths.iter() {
            if std::path::Path::new(path).exists() {
                tracing::info!("[SRManager] Loading config from: {}", path);
                return ConfigManager::from_toml_file(path);
            }
        }

        // 如果都找不到，返回错误
        Err(anyhow::anyhow!(
            "No models.toml config file found. Tried: {:?}",
            config_paths
        ))
    }

    // 开始超分辨率一个图片
    pub fn run_inference(&mut self, sr_info: SRInfo) -> Result<usize> {
        // 获取模型配置
        let model_config = self.config_manager.get_model(&sr_info.model_name)?;

        // 1. 仅获取图片尺寸，不加载完整图片数据
        let (width, height) = image::image_dimensions(&sr_info.input_path)?;
        let input_size = std::fs::metadata(&sr_info.input_path)?.len();

        let current_task_id = self.id_generator.generate_id();

        // 计算并存储元数据：使用配置中的 BORDER 和 OVERLAP
        let info = crate::pipeline::tiling_utils::TilingInfo::new_with_config(
            height as usize,
            width as usize,
            model_config.input_height,
            model_config.input_width,
            model_config.border,
            model_config.overlap,
        );
        self.task_metadata.insert(
            current_task_id,
            TaskMetaStruct {
                total_tiles: info.total_tiles(),
                input_size,
                input_width: width,
                input_height: height,
            },
        );

        // 2. 封装信息并发送给预处理器
        let preproc_info = PreprocessorInfo {
            task_id: current_task_id,
            input_path: sr_info.input_path.clone(),
            scale_factor: sr_info.scale_factor,
            model_name: sr_info.model_name.clone(),
            tile_width: model_config.input_width,
            tile_height: model_config.input_height,
            border: model_config.border,
            overlap: model_config.overlap,
        };
        self.preprocessor_tx
            .send(preproc_info)
            .map_err(|e| anyhow::anyhow!("Failed to send to preprocessor: {}", e))?;

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
