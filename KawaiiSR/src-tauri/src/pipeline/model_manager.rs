use crate::config::{ConfigManager, ModelConfig};
use anyhow::{Result, anyhow};
use dashmap::DashMap;
use ort::{
    execution_providers::DirectMLExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use std::sync::{Arc, Mutex};

/// 模型管理器：负责按需加载模型并管理 Session 缓存
pub struct ModelManager {
    /// 模型配置管理器
    config_manager: Arc<ConfigManager>,
    /// Session 缓存 (Key: model_name, Value: Session)
    sessions: DashMap<String, Arc<Mutex<Session>>>,
}

impl ModelManager {
    pub fn new(config_manager: Arc<ConfigManager>) -> Self {
        Self {
            config_manager,
            sessions: DashMap::new(),
        }
    }

    /// 获取模型 Session，如果缓存中有则直接返回，否则初始化
    pub fn get_session(&self, model_name: &str) -> Result<Arc<Mutex<Session>>> {
        // 1. 检查缓存
        if let Some(session) = self.sessions.get(model_name) {
            return Ok(session.value().clone());
        }

        // 2. 加载配置并初始化
        let model_config = self.config_manager.get_model(model_name)?;
        let session = Arc::new(Mutex::new(self.init_session(&model_config)?));

        // 3. 存入缓存
        self.sessions.insert(model_name.to_string(), session.clone());

        tracing::info!("[ModelManager] Model '{}' loaded and cached", model_name);
        Ok(session)
    }

    /// 初始化 ONNX Session
    fn init_session(&self, config: &ModelConfig) -> Result<Session> {
        let cwd = std::env::current_dir()?;
        let model_path = cwd.join(&config.file_path);

        tracing::info!("[ModelManager] Initializing model from: {:?}", model_path);
        if !model_path.exists() {
            return Err(anyhow!("Model file not found at {:?}", model_path));
        }

        // 尝试加载 Session，优先使用 DirectML
        let session = match Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_execution_providers([DirectMLExecutionProvider::default().build()])?
            .with_parallel_execution(false)?
            .with_memory_pattern(false)?
            .commit_from_file(&model_path)
        {
            Ok(s) => {
                tracing::info!("[ModelManager] ✓ Model '{}' loaded with DirectML", config.name);
                s
            }
            Err(e) => {
                tracing::error!(
                    "[ModelManager] ✗ DirectML failed for '{}': {:?}, falling back to CPU",
                    config.name, e
                );
                Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::All)?
                    .with_parallel_execution(false)?
                    .commit_from_file(&model_path)?
            }
        };

        Ok(session)
    }

    /// 清理特定模型的缓存
    pub fn evict_model(&self, model_name: &str) {
        if self.sessions.remove(model_name).is_some() {
            tracing::info!("[ModelManager] Model '{}' evicted from cache", model_name);
        }
    }

    /// 清理所有缓存
    pub fn clear_cache(&self) {
        self.sessions.clear();
        tracing::info!("[ModelManager] All models evicted from cache");
    }
}
