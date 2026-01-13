use std::{sync::Arc};
use crate::id_generator::IDGenerator;
use image::GenericImageView;
use ndarray::{Array3, Array4};
use crossbeam_channel::{Sender, Receiver, unbounded};
use tauri::{async_runtime::{Mutex}, image::Image};
use crate::pipeline::{ 
    image_stitcher::ImageStitcher, 
    image_tiler::{ImageTiler, TilerInfo}, 
    onnx_session::OnnxSession, 
    tensor_batcher::TensorBatcher
};
use anyhow::Result;
use crate::pipeline::image_meta::ImageMeta;

#[derive(Debug, Clone, Default)]
pub enum ModelName {
    #[default]
    KawaiiSR,
}




/// 用于对外表示的信息，
/// 可见范围：外部调用模块，SRManager
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

/// 对内表示的信息，可见范围：SRManager 以及 image_stitcher
/// 表示image_stticher工作结束，传回给SRManager的信息
pub struct ImageInfo {
    // 该图片任务的唯一标识符
    pub task_id: usize,
    // 图片的内容 (NCHW 格式)
    pub image_data: Array4<f32>,
    // 图片的元数据
    pub image_meta: Arc<ImageMeta>,
}


#[derive(Debug)]
pub struct SRManager {
    // 用于管理超分辨率管道的结构体
    inner: Arc<Mutex<SRManagerInner>>,
}


impl SRManager {
    pub async fn run_inference(&mut self, sr_info: SRInfo) -> Result<usize> {
        let mut inner = self.inner.lock().await;
        inner.run_inference(sr_info)
    }


    // 获取该图片的结果
    pub async fn get_result(&self, task_id: usize) -> Option<Arc<Image<'static>>> {
        let mut inner = self.inner.lock().await;
        inner.get_result(task_id)
    }

}

#[derive(Debug)]
pub struct SRManagerInner {
    // 将图片分割
    pub image_tiler: Arc<ImageTiler>,
    // 将图片合并成一个batch来运行
    pub tensor_batcher: Arc<Mutex<TensorBatcher>>,
    // Onnx推理
    pub onnx_session: Arc<Mutex<OnnxSession>>,
    // 图片恢复
    pub image_stitcher: Arc<Mutex<ImageStitcher>>,
    // 任务标识符生成器
    pub id_generator: Arc<IDGenerator>,
    // 向 Tiler 发送数据的通道 (manager 只维护这个)
    pub tiler_tx: Option<Sender<TilerInfo>>,
    // 接收从 stitcher 发回的图片结果的通道
    pub result_receiver: Receiver<ImageInfo>,
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

        // 创建各个 pipeline 组件，传入接收和发送通道
        let image_tiler = Arc::new(ImageTiler::new(manager_rx_tiler, tiler_tx_batcher));
        let tensor_batcher = Arc::new(Mutex::new(TensorBatcher::new(tiler_rx_batcher, batcher_tx_onnx)));
        let onnx_session = Arc::new(Mutex::new(OnnxSession::new(batcher_rx_onnx, onnx_tx_stitcher)));
        let image_stitcher = Arc::new(Mutex::new(ImageStitcher::new(onnx_rx_stitcher, stitcher_tx_manager)));

        Self {
            image_tiler,
            tensor_batcher,
            onnx_session,
            image_stitcher,
            id_generator: Arc::new(IDGenerator::default()),
            tiler_tx: Some(manager_tx_tiler),
            result_receiver: stitcher_rx_manager,
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
        });


        // 2. 将其变成ndarray [3, H, W] f32 归一化 (CHW)
        let mut array = Array3::<f32>::zeros((3, height as usize, width as usize));
        for (x, y, pixel) in rgb_img.enumerate_pixels() {
            for c in 0..3 {
                array[[c, y as usize, x as usize]] = pixel[c] as f32 / 255.0;
            }
        }

        let current_task_id = self.id_generator.generate_id();

        // 3. 封装信息并发送给 image_tiler
        if let Some(tx) = &self.tiler_tx {
            let tiler_info = TilerInfo {
                task_id: current_task_id,
                image_data: array,
                image_meta: image_meta,
            };
            tx.send(tiler_info).map_err(|e| anyhow::anyhow!("Failed to send to tiler: {}", e))?;
        } else {
            return Err(anyhow::anyhow!("Tiler channel not initialized"));
        }

        Ok(current_task_id)
    }


    // 获取该图片的结果
    pub fn get_result(&mut self, _task_id: usize) -> Option<Arc<Image<'static>>> {
        // 从image_meta_storage中获取结果
        None
    }
    
}

