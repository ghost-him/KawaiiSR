use std::sync::Arc;

use crate::sr_manager::SRManager;

// 用于管理应用程序的全局状态
#[derive(Default)]
pub struct AppState {
    // 表示一个管道
    pub sr_pipline: Option<Arc<SRManager>>,




}