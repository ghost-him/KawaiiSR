<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { 
  NButton, NCard, NSpace, NText, useMessage, NSelect, NInputNumber, NForm, NFormItem, NSwitch,
  NInput, NInputGroup
} from "naive-ui";
import { useTasks } from "../composables/useTasks";
import type { Task } from "../types";

const emit = defineEmits(['taskCreated']);
const message = useMessage();
const { addTask, selectTask, getAvailableModels, getDefaultModel } = useTasks();

interface TaskMetaStruct {
  total_tiles: number;
  input_size: number;
  input_width: number;
  input_height: number;
}

interface ModelConfig {
  name: string;
  file_path: string;
  input_width: number;
  input_height: number;
  overlap: number;
  border: number;
  batch_size: number;
  description: string;
}

const availableModels = ref<{ label: string, value: string }[]>([]);
const selectedModel = ref<string>("");
const selectedModelDescription = ref<string>("");
const scaleFactor = ref<number>(2);

// 自定义切块参数
const useCustomTiling = ref(false);
const customOverlap = ref(32);
const customBorder = ref(64);
const modelDefaultOverlap = ref(32);
const modelDefaultBorder = ref(64);

onMounted(async () => {
  const models = await getAvailableModels();
  availableModels.value = models.map(m => ({ label: m, value: m }));
  
  if (models.length > 0) {
    const defaultModel = await getDefaultModel();
    selectedModel.value = defaultModel || models[0];
  }
});

// 当模型改变时，更新默认参数
watch(selectedModel, async (newModel) => {
  if (newModel) {
    console.log("Selected model changed to:", newModel);
    try {
      const config = await invoke<ModelConfig>("get_model_config", { modelName: newModel });
      console.log("Fetched model config:", config);
      modelDefaultOverlap.value = config.overlap;
      modelDefaultBorder.value = config.border;
      selectedModelDescription.value = config.description;
      
      // 如果没有开启自定义，则同步更新显示的值
      if (!useCustomTiling.value) {
        customOverlap.value = config.overlap;
        customBorder.value = config.border;
      }
    } catch (err) {
      console.error("Failed to fetch model config:", err);
    }
  }
});

const autoSaveDir = ref<string | null>(null);

async function pickAutoSaveDir() {
  const selected = await open({
    directory: true,
    multiple: false,
  });
  if (selected && !Array.isArray(selected)) {
    autoSaveDir.value = selected;
  }
}

async function createSingleTask(inputPath: string, setActive: boolean = true) {
  const filename = inputPath.split(/[\\/]/).pop() || "image.png";
  const scale = scaleFactor.value;
  const modelName = selectedModel.value;

  let outputPath = null;
  if (autoSaveDir.value) {
    const sep = (autoSaveDir.value.includes("\\") || inputPath.includes("\\")) ? "\\" : "/";
    outputPath = `${autoSaveDir.value}${sep}${filename}`;
    // Force PNG
    if (!outputPath.toLowerCase().endsWith(".png")) {
      outputPath = outputPath.replace(/\.[^/.]+$/, "") + ".png";
    }
  }
  
  try {
    const id = await invoke<number>("run_super_resolution", {
      inputPath: inputPath,
      modelName: modelName,
      outputPath: outputPath,
      overlap: useCustomTiling.value ? customOverlap.value : null,
      border: useCustomTiling.value ? customBorder.value : null
    });
    
    const metadata = await invoke<TaskMetaStruct>("get_task_metadata", { taskId: id });
    
    const newTask: Task = {
      id,
      inputPath,
      filename,
      scaleFactor: scale,
      modelName: modelName,
      status: 'processing',
      progress: 0, 
      completedTiles: 0,
      totalTiles: metadata.total_tiles,
      inputSize: metadata.input_size,
      inputWidth: metadata.input_width,
      inputHeight: metadata.input_height,
      startTime: Date.now()
    };
    
    addTask(newTask, setActive);
    if (setActive) {
      selectTask(id);
    }
    emit('taskCreated', id);
    
  } catch (err) {
    console.error("Failed to start task for " + inputPath + ":", err);
    message.error(`无法启动任务 (${filename}): ${err}`);
  }
}

async function startNewTask() {
  if (!selectedModel.value) {
    message.warning("请先选择一个模型");
    return;
  }

  const selected = await open({
    multiple: true,
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }]
  });
  
  if (selected && Array.isArray(selected)) {
    // 循环处理选中的多张图片
    for (let i = 0; i < selected.length; i++) {
      // 第一张设置为活跃任务，后续的不自动跳转
      await createSingleTask(selected[i], i === 0);
    }
    if (selected.length > 0) {
      message.success(`成功启动 ${selected.length} 个任务`);
    }
  } else if (selected) {
    // 兼容某些情况下的单选返回
    await createSingleTask(selected as string, true);
  }
}

async function startBatchFolderTask() {
  if (!selectedModel.value) {
    message.warning("请先选择一个模型");
    return;
  }

  if (!autoSaveDir.value) {
    message.warning("建议先选择“自动保存”目录，否则处理完成后的图片将仅保存在内存中，需要手动一张张保存。");
  }

  const folder = await open({
    directory: true,
    multiple: false,
  });

  if (folder && typeof folder === 'string') {
    try {
      const images = await invoke<string[]>("list_images_in_folder", { path: folder });
      if (images.length === 0) {
        message.info("该文件夹中没有找到匹配的图片格式 (png, jpg, jpeg, webp)");
        return;
      }

      message.loading(`正在为 ${images.length} 张图片创建任务...`);
      
      for (let i = 0; i < images.length; i++) {
        await createSingleTask(images[i], i === 0);
      }
      
      message.success(`成功从文件夹中启动 ${images.length} 个任务`);
    } catch (err) {
      console.error("Failed to list folder:", err);
      message.error("无法读取文件夹内容: " + err);
    }
  }
}
</script>

<template>
  <div class="createTaskPage">
    <h1 style="margin-bottom: 40px; margin-top: 0; font-size: 24px;">创建新任务</h1>
    <n-card :bordered="false" style="border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
      <n-space vertical size="large">
        <n-text depth="3">
          配置您的超分辨率参数，选择合适的模型并设置缩放倍数。
        </n-text>
        
        <n-form label-placement="left" label-width="100" style="margin-top: 20px;">
          <n-form-item label="选择模型">
            <div class="model-selection-wrapper">
              <n-select 
                v-model:value="selectedModel" 
                :options="availableModels" 
                placeholder="加载中..."
                class="model-select"
              />
              <n-text v-if="selectedModelDescription" class="model-description">
                {{ selectedModelDescription }}
              </n-text>
            </div>
          </n-form-item>
          
          <!--当前不支持选择缩放倍数-->
          <!-- <n-form-item label="缩放倍数">
            <n-input-number 
              v-model:value="scaleFactor" 
              :min="1" 
              :max="4" 
              :step="1"
              style="width: 100%"
            />
          </n-form-item> -->

          <n-form-item label="自动保存">
            <n-input-group>
              <n-input 
                v-model:value="autoSaveDir" 
                placeholder="任务完成后自动保存到此目录 (可选)" 
                readonly
                @click="pickAutoSaveDir"
              />
              <n-button type="primary" ghost @click="pickAutoSaveDir">
                选择目录
              </n-button>
              <n-button v-if="autoSaveDir" @click="autoSaveDir = null">
                清除
              </n-button>
            </n-input-group>
          </n-form-item>

          <n-form-item label="高级参数">
            <n-space align="center">
              <n-switch v-model:value="useCustomTiling" />
              <n-text depth="3">自定义切块 (Padding & Border)</n-text>
            </n-space>
          </n-form-item>

          <transition name="fade">
            <div v-if="useCustomTiling" style="background: rgba(0,0,0,0.02); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
              <n-space vertical>
                <n-form-item label="Overlap (Padding)" label-placement="left" :show-feedback="false">
                  <n-input-number v-model:value="customOverlap" :min="0" :max="128" style="width: 100%">
                    <template #suffix>px</template>
                  </n-input-number>
                </n-form-item>
                <n-text depth="3" style="font-size: 12px; margin-bottom: 10px;">
                  模型默认值: {{ modelDefaultOverlap }}px. 增加重叠可以减少拼接处的接缝感。
                </n-text>
                
                <n-form-item label="Border (Extra)" label-placement="left" :show-feedback="false">
                  <n-input-number v-model:value="customBorder" :min="0" :max="256" style="width: 100%">
                    <template #suffix>px</template>
                  </n-input-number>
                </n-form-item>
                <n-text depth="3" style="font-size: 12px;">
                  模型默认值: {{ modelDefaultBorder }}px. 必须大于等于 Overlap。
                </n-text>
              </n-space>
            </div>
          </transition>
        </n-form>

        <n-space vertical size="medium" style="margin-top: 20px;">
          <n-button 
            type="primary" 
            size="large" 
            @click="startNewTask" 
            block 
            style="height: 50px; font-weight: bold;"
            :disabled="!selectedModel"
          >
            选择图片并开始处理
          </n-button>
          
          <n-button 
            type="info" 
            size="large" 
            ghost
            @click="startBatchFolderTask" 
            block 
            style="height: 50px; font-weight: bold;"
            :disabled="!selectedModel"
          >
            处理整个文件夹
          </n-button>
        </n-space>
      </n-space>
    </n-card>
  </div>
</template>

<style scoped>
.createTaskPage {
  width: 100%;
}

.model-selection-wrapper {
  display: flex;
  align-items: center;
  width: 100%;
  gap: 16px;
}

.model-select {
  width: 200px;
  flex-shrink: 0;
}

.model-description {
  flex: 1;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 当宽度不足时，隐藏描述或调整布局 */
@media (max-width: 800px) {
  .model-description {
    display: none;
  }
  .model-select {
    width: 100%;
  }
}

.fade-enter-active, .fade-leave-active {
  transition: opacity 0.3s, transform 0.3s;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>
