use super::model_config::{ModelConfig, NormalizationConfig, NormalizationRange};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use std::sync::Arc;

/// 配置管理器：负责加载、验证和管理模型配置
pub struct ConfigManager {
    /// 所有已加载的模型配置（Key: 模型名称）
    models: Arc<DashMap<String, Arc<ModelConfig>>>,
    /// 默认模型名称
    default_model_name: String,
}

impl ConfigManager {
    /// 从 TOML 配置文件加载所有模型配置
    /// 配置文件格式示例见 models.toml
    pub fn from_toml_file(file_path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| anyhow!("Failed to read config file {}: {}", file_path, e))?;

        Self::from_toml_string(&content)
    }

    /// 从 TOML 字符串加载配置
    pub fn from_toml_string(content: &str) -> Result<Self> {
        // 解析 TOML
        let config: toml::Value =
            toml::from_str(content).map_err(|e| anyhow!("Failed to parse TOML config: {}", e))?;

        let models = Arc::new(DashMap::new());
        let mut default_model_name = String::new();

        // 1. 先读取 defaults 部分
        if let Some(defaults) = config.get("defaults") {
            if let Some(name) = defaults.get("model_name").and_then(|v| v.as_str()) {
                default_model_name = name.to_string();
            }
        }

        // 2. 读取 models 部分
        if let Some(models_table) = config.get("models").and_then(|v| v.as_table()) {
            for (model_name, model_config) in models_table.iter() {
                match Self::parse_model_config(model_name, model_config) {
                    Ok(config) => {
                        models.insert(model_name.to_string(), Arc::new(config));
                        tracing::info!("[ConfigManager] Loaded model: {}", model_name);
                    }
                    Err(e) => {
                        tracing::error!(
                            "[ConfigManager] Failed to load model '{}': {}",
                            model_name,
                            e
                        );
                        return Err(e);
                    }
                }
            }
        } else {
            return Err(anyhow!("No 'models' section found in config"));
        }

        // 3. 如果没有指定默认模型，就用第一个加载的模型
        if default_model_name.is_empty() {
            if let Some(entry) = models.iter().next() {
                default_model_name = entry.key().clone();
                tracing::warn!(
                    "[ConfigManager] No default model specified, using: {}",
                    default_model_name
                );
            } else {
                return Err(anyhow!("No models found in config"));
            }
        }

        // 4. 验证默认模型存在
        if !models.contains_key(&default_model_name) {
            return Err(anyhow!(
                "Default model '{}' not found in config",
                default_model_name
            ));
        }

        tracing::info!(
            "[ConfigManager] Configuration loaded with {} models, default: {}",
            models.len(),
            default_model_name
        );

        Ok(Self {
            models,
            default_model_name,
        })
    }

    /// 解析单个模型的配置
    fn parse_model_config(name: &str, config: &toml::Value) -> Result<ModelConfig> {
        let file_path = config
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'file_path' for model '{}'", name))?
            .to_string();

        let input_width = config
            .get("input_width")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| anyhow!("Missing 'input_width' for model '{}'", name))?
            as usize;

        let input_height = config
            .get("input_height")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| anyhow!("Missing 'input_height' for model '{}'", name))?
            as usize;

        let overlap = config
            .get("overlap")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| anyhow!("Missing 'overlap' for model '{}'", name))?
            as usize;

        let border = config
            .get("border")
            .and_then(|v| v.as_integer())
            .unwrap_or(64) as usize; // 默认 border = 64

        let scale = config
            .get("scale")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| anyhow!("Missing 'scale' for model '{}'", name))?
            as u32;

        let input_node = config
            .get("input_node")
            .and_then(|v| v.as_str())
            .unwrap_or("input")
            .to_string();

        let output_node = config
            .get("output_node")
            .and_then(|v| v.as_str())
            .unwrap_or("output")
            .to_string();

        let description = config
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // 解析归一化配置
        let normalization = Self::parse_normalization_config(name, config)?;

        let model_config = ModelConfig {
            name: name.to_string(),
            file_path,
            input_width,
            input_height,
            overlap,
            border,
            scale,
            input_node,
            output_node,
            normalization,
            description,
        };

        // 验证配置（包括自动调整 border）
        model_config.validate()
    }

    /// 解析归一化配置
    fn parse_normalization_config(name: &str, config: &toml::Value) -> Result<NormalizationConfig> {
        let normalization_table = config.get("normalization").and_then(|v| v.as_table());

        if let Some(norm) = normalization_table {
            let range_str = norm
                .get("range")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("Missing 'normalization.range' for model '{}'", name))?;

            let range = match range_str {
                "zero_to_one" => NormalizationRange::ZeroToOne,
                "minus_one_to_one" => NormalizationRange::MinusOneToOne,
                "zero_to_255" => NormalizationRange::ZeroTo255,
                _ => {
                    return Err(anyhow!(
                        "Invalid normalization range '{}' for model '{}'",
                        range_str,
                        name
                    ))
                }
            };

            let mean = norm.get("mean").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_float())
                    .map(|f| f as f32)
                    .collect()
            });

            let std = norm.get("std").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_float())
                    .map(|f| f as f32)
                    .collect()
            });

            Ok(NormalizationConfig { range, mean, std })
        } else {
            // 默认归一化配置
            Ok(NormalizationConfig {
                range: NormalizationRange::ZeroToOne,
                mean: None,
                std: None,
            })
        }
    }

    /// 获取指定模型的配置
    pub fn get_model(&self, model_name: &str) -> Result<Arc<ModelConfig>> {
        self.models
            .get(model_name)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| anyhow!("Model '{}' not found", model_name))
    }

    /// 获取默认模型的配置
    pub fn get_default_model(&self) -> Result<Arc<ModelConfig>> {
        self.get_model(&self.default_model_name)
    }

    /// 获取所有可用的模型名称列表
    pub fn list_models(&self) -> Vec<String> {
        self.models
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// 获取默认模型名称
    pub fn default_model_name(&self) -> &str {
        &self.default_model_name
    }

    /// 获取已加载的模型数量
    pub fn model_count(&self) -> usize {
        self.models.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_toml_string() {
        let toml_content = r#"
[models.test_model]
file_path = "test.onnx"
input_width = 128
input_height = 128
overlap = 32
border = 64
scale = 2
input_node = "input"
output_node = "output"
[models.test_model.normalization]
range = "zero_to_one"

[defaults]
model_name = "test_model"
        "#;

        let manager = ConfigManager::from_toml_string(toml_content).unwrap();
        assert_eq!(manager.model_count(), 1);
        assert_eq!(manager.default_model_name(), "test_model");
        assert!(manager.get_model("test_model").is_ok());
    }

    #[test]
    fn test_list_models() {
        let toml_content = r#"
[models.model_a]
file_path = "a.onnx"
input_width = 128
input_height = 128
overlap = 32
scale = 2
input_node = "input"
output_node = "output"
[models.model_a.normalization]
range = "zero_to_one"

[models.model_b]
file_path = "b.onnx"
input_width = 256
input_height = 256
overlap = 32
scale = 4
input_node = "input"
output_node = "output"
[models.model_b.normalization]
range = "zero_to_one"
std = [0.229, 0.224, 0.225]

[defaults]
model_name = "model_a"
        "#;

        let manager = ConfigManager::from_toml_string(toml_content).unwrap();
        let models = manager.list_models();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"model_a".to_string()));
        assert!(models.contains(&"model_b".to_string()));
        if let Ok(model) = manager.get_model("model_b") {
            assert_eq!(model.input_width, 256);
            assert_eq!(model.scale, 4);
            assert_eq!(model.normalization.range, NormalizationRange::ZeroToOne);
            assert_eq!(model.normalization.std, Some(vec![0.229, 0.224, 0.225]));
            assert!(model.normalization.mean.is_none());
        } else {
            panic!("Model 'model_b' should exist");
        }
    }
}
