// 保存图片的

/// 表示一个图片的元数据
pub struct ImageMeta {
    // 图片原始宽度
    pub original_width: usize,
    // 图片原始高度
    pub original_height: usize,
    // 放大的倍数（当前没用，由模型训练时的参数决定）
    pub scale_factor: u32,
}
