use crate::pipeline::image_meta::ImageMeta;
use crate::pipeline::image_stitcher::StitcherInfo;
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashSet;
use ndarray::Array4;
use ort::{
    execution_providers::DirectMLExecutionProvider,
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::sync::Arc;

pub struct OnnxSessionInfo {
    // 可能会将不同任务的多个切片一起送入ONNX进行推理，所以要区分不同的batch上是哪一个任务的切片
    pub task_id: Vec<usize>,
    pub tile_index: Vec<usize>,
    // 进行推理的切片数据 (NCHW 格式)
    pub batch_data: Array4<f32>,
    // 每个切片对应的图片元数据
    pub image_meta: Vec<Arc<ImageMeta>>,
}

pub struct OnnxSession {
    _handle: std::thread::JoinHandle<()>,
}

impl OnnxSession {
    pub fn new(
        batcher_rx: Receiver<OnnxSessionInfo>,
        stitcher_tx: Sender<StitcherInfo>,
        cancelled_tasks: Arc<DashSet<usize>>,
    ) -> Self {
        let mut inner = OnnxSessionInner {
            batcher_rx,
            stitcher_tx,
            cancelled_tasks,
            session: None,
        };

        let handle = std::thread::spawn(move || {
            inner.execute();
        });

        Self { _handle: handle }
    }
}

struct OnnxSessionInner {
    batcher_rx: Receiver<OnnxSessionInfo>,
    stitcher_tx: Sender<StitcherInfo>,
    cancelled_tasks: Arc<DashSet<usize>>,
    session: Option<Session>,
}

impl OnnxSessionInner {
    fn execute(&mut self) {
        tracing::info!("[OnnxSession] Started");

        // 1. 初始化 ONNX 模型
        if let Err(e) = self.initialize_model() {
            tracing::error!("[OnnxSession] Failed to initialize model: {:?}", e);
            return;
        }

        // 2. 接收数据并进行推理
        while let Ok(onnx_info) = self.batcher_rx.recv() {
            // 检查该 batch 中的所有任务是否都已取消。
            // 只要有一个任务没取消，我们就得跑这个 batch（因为切片是混在一起的）。
            // 但如果全部取消了，可以直接跳过。
            if onnx_info.task_id.iter().all(|id| self.cancelled_tasks.contains(id)) {
                tracing::info!("[OnnxSession] Skipping batch as all tasks are cancelled");
                continue;
            }

            tracing::info!(
                "[OnnxSession] Processing batch with {} tiles, shape: {:?}",
                onnx_info.task_id.len(),
                onnx_info.batch_data.shape()
            );

            if let Err(e) = self.process_batch(onnx_info) {
                tracing::error!("[OnnxSession] Failed to process batch: {:?}", e);
                // 继续处理下一批，不中断整个流程
            }
        }

        tracing::info!("[OnnxSession] Stopped");
    }

    fn initialize_model(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let cwd = std::env::current_dir()?;
        let mut model_path = cwd.join("onnx/kawaii_sr_128x128.onnx");
        if !model_path.exists() {
            model_path = cwd.join("src-tauri/onnx/kawaii_sr_128x128.onnx");
        }

        tracing::info!("[OnnxSession] Loading model from: {:?}", model_path);
        if !model_path.exists() {
            return Err(format!("Model not found at {:?}", model_path).into());
        }

        let session = {
            tracing::info!("[OnnxSession] Attempting to load with DirectML...");
            match Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::All)?
                .with_execution_providers([DirectMLExecutionProvider::default().build()])?
                .with_parallel_execution(false)?
                .with_memory_pattern(false)?
                .commit_from_file(&model_path)
            {
                Ok(s) => {
                    tracing::info!("[OnnxSession] ✓ Model loaded with DirectML");
                    s
                }
                Err(e) => {
                    tracing::error!(
                        "[OnnxSession] ✗ DirectML failed: {:?}, falling back to CPU",
                        e
                    );
                    Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::All)?
                        .with_parallel_execution(false)?
                        .commit_from_file(&model_path)?
                }
            }
        };

        self.session = Some(session);
        Ok(())
    }

    fn process_batch(
        &mut self,
        onnx_info: OnnxSessionInfo,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let session = self
            .session
            .as_mut()
            .ok_or("ONNX session not initialized")?;

        // 运行推理
        // 确保输入是标准布局 (Standard Layout)，并避免在已经是标准布局时进行额外拷贝
        let input_tensor =
            Value::from_array(onnx_info.batch_data.as_standard_layout().into_owned())?;
        let outputs = session.run(inputs!["input" => input_tensor])?;

        let output_value = outputs
            .get("output")
            .ok_or("Failed to get output 'output'")?;
        let (output_shape, output_data) = output_value.try_extract_tensor::<f32>()?;

        tracing::info!(
            "[OnnxSession] Inference completed, output shape: {:?}",
            output_shape
        );

        // 构建输出数据
        let out_n = output_shape[0] as usize;
        let out_c = output_shape[1] as usize;
        let out_h = output_shape[2] as usize;
        let out_w = output_shape[3] as usize;

        let stitched_data =
            Array4::from_shape_vec((out_n, out_c, out_h, out_w), output_data.to_vec())?;

        // 发送到 Stitcher
        let stitcher_info = StitcherInfo {
            task_id: onnx_info.task_id,
            tile_index: onnx_info.tile_index,
            stitched_data,
            image_meta: onnx_info.image_meta,
        };

        self.stitcher_tx
            .send(stitcher_info)
            .map_err(|e| format!("Failed to send to stitcher: {}", e))?;

        Ok(())
    }
}
