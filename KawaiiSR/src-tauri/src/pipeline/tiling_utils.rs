pub const BORDER: usize = 64;
pub const TILE_SIZE: usize = 128;
pub const OVERLAP: usize = 32;

pub struct TilingInfo {
    pub original_h: usize,
    pub original_w: usize,
    pub border_h: usize, // 添加 2*BORDER 后的尺寸
    pub border_w: usize,
    pub padded_h: usize, // 向上对齐到步长倍数后的填充尺寸
    pub padded_w: usize,
    pub rows: usize,
    pub cols: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub tile_h: usize,  // 实际使用的切块高度
    pub tile_w: usize,  // 实际使用的切块宽度
    pub border: usize,  // 实际使用的边界大小
    pub overlap: usize, // 实际使用的重叠大小
}

impl TilingInfo {
    pub fn new(h: usize, w: usize) -> Self {
        Self::new_with_config(h, w, TILE_SIZE, TILE_SIZE, BORDER, OVERLAP)
    }

    /// 使用自定义的参数创建 TilingInfo
    pub fn new_with_config(
        h: usize,
        w: usize,
        tile_h: usize,
        tile_w: usize,
        border: usize,
        overlap: usize,
    ) -> Self {
        // 1. 添加边界后的尺寸
        let border_h = h + border * 2;
        let border_w = w + border * 2;

        // 2. 步长 = tile_size - overlap
        let stride_h = tile_h - overlap;
        let stride_w = tile_w - overlap;

        // 3. 计算需要的 padding 使得步进能覆盖所有像素
        let rows = if border_h <= tile_h {
            1
        } else {
            ((border_h - tile_h + stride_h - 1) / stride_h) + 1
        };

        let cols = if border_w <= tile_w {
            1
        } else {
            ((border_w - tile_w + stride_w - 1) / stride_w) + 1
        };

        let padded_h = (rows - 1) * stride_h + tile_h;
        let padded_w = (cols - 1) * stride_w + tile_w;

        Self {
            original_h: h,
            original_w: w,
            border_h,
            border_w,
            padded_h,
            padded_w,
            rows,
            cols,
            stride_h,
            stride_w,
            tile_h,
            tile_w,
            border,
            overlap,
        }
    }

    /// 获取第 i 个 tile 的起始坐标 (y, x)
    pub fn get_tile_start(&self, index: usize) -> (usize, usize) {
        let row = index / self.cols;
        let col = index % self.cols;
        (row * self.stride_h, col * self.stride_w)
    }

    /// 获取总分块数
    pub fn total_tiles(&self) -> usize {
        self.rows * self.cols
    }
}
