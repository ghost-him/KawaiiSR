use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// 模型配置，包含模型路径、输入尺寸和切块参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// 模型名称
    pub name: String,
    /// 模型文件路径（相对于当前工作目录）
    pub file_path: String,
    /// 模型期望的输入宽度
    pub input_width: usize,
    /// 模型期望的输入高度
    pub input_height: usize,
    /// 切块时的默认重叠大小
    pub overlap: usize,
    /// 边界填充大小（边框）
    pub border: usize,
}

impl ModelConfig {
    /// 验证配置的有效性
    /// 如果 border < overlap，自动调整 border = overlap
    pub fn validate(mut self) -> Result<Self> {
        // 验证必要的字段不为空
        if self.name.is_empty() {
            return Err(anyhow!("Model name cannot be empty"));
        }
        if self.file_path.is_empty() {
            return Err(anyhow!("Model file path cannot be empty"));
        }

        // 验证尺寸合理性
        if self.input_width == 0 || self.input_height == 0 {
            return Err(anyhow!(
                "Model input dimensions must be greater than 0: {}x{}",
                self.input_width,
                self.input_height
            ));
        }

        // 验证 overlap 合理性
        if self.overlap == 0 {
            return Err(anyhow!("Overlap must be greater than 0"));
        }

        // 核心逻辑：如果 border < overlap，自动调整 border = overlap
        if self.border < self.overlap {
            tracing::warn!(
                "[ModelConfig] Model '{}': border {} < overlap {}, auto-adjusting border to {}",
                self.name,
                self.border,
                self.overlap,
                self.overlap
            );
            self.border = self.overlap;
        }

        tracing::info!(
            "[ModelConfig] Model '{}' validated: {}x{}, overlap={}, border={}",
            self.name,
            self.input_width,
            self.input_height,
            self.overlap,
            self.border
        );

        Ok(self)
    }

    /// 验证模型文件是否存在
    pub fn verify_file_exists(&self) -> Result<()> {
        let path = std::path::Path::new(&self.file_path);
        if !path.exists() {
            return Err(anyhow!("Model file not found: {}", self.file_path));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_border_auto_adjust() {
        let config = ModelConfig {
            name: "test".to_string(),
            file_path: "test.onnx".to_string(),
            input_width: 128,
            input_height: 128,
            overlap: 32,
            border: 16, // border < overlap
        };

        let validated = config.validate().unwrap();
        assert_eq!(
            validated.border, 32,
            "Border should be auto-adjusted to overlap value"
        );
    }

    #[test]
    fn test_validate_empty_name() {
        let config = ModelConfig {
            name: "".to_string(),
            file_path: "test.onnx".to_string(),
            input_width: 128,
            input_height: 128,
            overlap: 32,
            border: 64,
        };

        assert!(config.validate().is_err());
    }
}
