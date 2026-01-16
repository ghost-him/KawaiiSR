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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImageType {
    RGB,            // 该tile是属于RGB的
    RGBA(bool),     // 该tile是属于RGBA，bool表示是否是Alpha通道的tile
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TaskType {
    Image(ImageType),   // 如果是一个图片，那么会指出来该图片是RGB还是RGBA的
    Video(usize),       // 如果是一个视频，那么会指出来该视频的帧数（即，当前的tile是第几帧的tile）
}