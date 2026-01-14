use std::sync::Arc;

use crate::sr_manager::SRManager;

// 用于管理应用程序的全局状态
pub struct AppState {
    // 表示一个管道
    pub sr_pipline: Arc<SRManager>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            sr_pipline: Arc::new(SRManager::default()),
        }
    }
}