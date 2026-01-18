use crate::pipeline::image_stitcher::StitcherInfo;
use crate::pipeline::model_manager::ModelManager;
use crate::pipeline::task_meta::ImageMeta;
use crate::pipeline::task_meta::TaskType;
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashSet;
use ndarray::Array4;
use ort::{inputs, value::Value};
use std::sync::Arc;

pub struct OnnxSessionInfo {
    // 可能会将不同任务的多个切片一起送入ONNX进行推理，所以要区分不同的batch上是哪一个任务的切片
    pub task_id: Vec<usize>,
    pub tile_index: Vec<usize>,
    pub task_type: Vec<TaskType>,
    // 进行推理的切片数据 (NCHW 格式)
    pub batch_data: Array4<f32>,
    // 每个切片对应的图片元数据
    pub image_meta: Vec<Arc<ImageMeta>>,
    // 该 batch 使用的模型名称（由 TensorBatcher 确保同一 batch 物理上属于同一模型）
    pub model_name: String,
}

pub struct OnnxSession {
    _handle: std::thread::JoinHandle<()>,
}

impl OnnxSession {
    pub fn new(
        batcher_rx: Receiver<OnnxSessionInfo>,
        stitcher_tx: Sender<StitcherInfo>,
        cancelled_tasks: Arc<DashSet<usize>>,
        model_manager: Arc<ModelManager>,
    ) -> Self {
        let mut inner = OnnxSessionInner {
            batcher_rx,
            stitcher_tx,
            cancelled_tasks,
            model_manager,
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
    model_manager: Arc<ModelManager>,
}

impl OnnxSessionInner {
    fn execute(&mut self) {
        tracing::info!("[OnnxSession] Started");

        // 接收数据并进行推理
        while let Ok(onnx_info) = self.batcher_rx.recv() {
            // 检查该 batch 中的所有任务是否都已取消。
            if onnx_info
                .task_id
                .iter()
                .all(|id| self.cancelled_tasks.contains(id))
            {
                tracing::info!("[OnnxSession] Skipping batch as all tasks are cancelled");
                continue;
            }

            tracing::info!(
                "[OnnxSession] Processing batch with {} tiles for model '{}', shape: {:?}",
                onnx_info.task_id.len(),
                onnx_info.model_name,
                onnx_info.batch_data.shape()
            );

            if let Err(e) = self.process_batch(onnx_info) {
                tracing::error!("[OnnxSession] Failed to process batch: {:?}", e);
            }
        }

        tracing::info!("[OnnxSession] Stopped");
    }

    fn process_batch(
        &mut self,
        onnx_info: OnnxSessionInfo,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 从管理器获取对应的模型 Session 的 Mutex
        let session_mutex = self.model_manager.get_session(&onnx_info.model_name)?;
        let model_config = self.model_manager.get_config(&onnx_info.model_name)?;
        let mut session = session_mutex
            .lock()
            .map_err(|e| format!("Failed to lock session: {}", e))?;

        // 1. 将 ndarray 转换为 ONNX Value (确保标准布局)
        let input_tensor =
            Value::from_array(onnx_info.batch_data.as_standard_layout().into_owned())?;

        // 2. 运行推理，使用动态的输入输出节点名称
        let input_name = &model_config.input_node;
        let output_name = &model_config.output_node;
        let outputs = session.run(inputs![input_name => input_tensor])?;

        let output_value = outputs
            .get(output_name)
            .ok_or(format!("Failed to get output '{}'", output_name))?;
        let (output_shape, output_data) = output_value.try_extract_tensor::<f32>()?;

        // 3. 构建输出张量
        let out_n = output_shape[0] as usize;
        let out_c = output_shape[1] as usize;
        let out_h = output_shape[2] as usize;
        let out_w = output_shape[3] as usize;

        let stitched_data =
            Array4::from_shape_vec((out_n, out_c, out_h, out_w), output_data.to_vec())?;

        // 4. 发送到 Stitcher
        let stitcher_info = StitcherInfo {
            task_ids: onnx_info.task_id,
            tile_indexs: onnx_info.tile_index,
            task_types: onnx_info.task_type,
            stitched_data,
            image_metas: onnx_info.image_meta,
        };

        if let Err(e) = self.stitcher_tx.send(stitcher_info) {
            tracing::error!("[OnnxSession] Failed to send to stitcher: {}", e);
        }

        Ok(())
    }
}
