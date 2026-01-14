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
    pub stride: usize,
}

impl TilingInfo {
    pub fn new(h: usize, w: usize) -> Self {
        // 1. 添加 64px 边缘后的尺寸
        let border_h = h + BORDER * 2;
        let border_w = w + BORDER * 2;
        
        // 2. 步长 = TILE_SIZE - OVERLAP
        let stride = TILE_SIZE - OVERLAP;
        
        // 3. 计算需要的 padding 使得步进能覆盖所有像素
        // 也就是满足 (rows - 1) * stride + TILE_SIZE >= border_h
        // 即 rows - 1 >= (border_h - TILE_SIZE + stride - 1) / stride
        let rows = if border_h <= TILE_SIZE {
            1
        } else {
            ((border_h - TILE_SIZE + stride - 1) / stride) + 1
        };
        
        let cols = if border_w <= TILE_SIZE {
            1
        } else {
            ((border_w - TILE_SIZE + stride - 1) / stride) + 1
        };
        
        let padded_h = (rows - 1) * stride + TILE_SIZE;
        let padded_w = (cols - 1) * stride + TILE_SIZE;
        
        Self {
            original_h: h,
            original_w: w,
            border_h,
            border_w,
            padded_h,
            padded_w,
            rows,
            cols,
            stride,
        }
    }

    /// 获取第 i 个 tile 的起始坐标 (y, x)
    pub fn get_tile_start(&self, index: usize) -> (usize, usize) {
        let row = index / self.cols;
        let col = index % self.cols;
        (row * self.stride, col * self.stride)
    }

    /// 获取总分块数
    pub fn total_tiles(&self) -> usize {
        self.rows * self.cols
    }
}
