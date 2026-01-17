<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { 
  NButton, NCard, NSpace, NText, useMessage, NSelect, NInputNumber, NForm, NFormItem
} from "naive-ui";
import { useTasks } from "../composables/useTasks";
import type { Task } from "../types";

const emit = defineEmits(['taskCreated']);
const message = useMessage();
const { addTask, selectTask, getAvailableModels, getDefaultModel } = useTasks();

interface TaskMetaStruct {
  total_tiles: number;
}

const availableModels = ref<{ label: string, value: string }[]>([]);
const selectedModel = ref<string>("");
const scaleFactor = ref<number>(2);

onMounted(async () => {
  const models = await getAvailableModels();
  availableModels.value = models.map(m => ({ label: m, value: m }));
  
  if (models.length > 0) {
    const defaultModel = await getDefaultModel();
    selectedModel.value = defaultModel || models[0];
  }
});

async function startNewTask() {
  if (!selectedModel.value) {
    message.warning("请先选择一个模型");
    return;
  }

  const selected = await open({
    multiple: false,
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }]
  });
  
  if (selected && !Array.isArray(selected)) {
    const inputPath = selected;
    const filename = inputPath.split(/[\\/]/).pop() || "image.png";
    const scale = scaleFactor.value;
    const modelName = selectedModel.value;
    
    try {
      const id = await invoke<number>("run_super_resolution", {
        inputPath: inputPath,
        modelName: modelName,
        scaleFactor: scale
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
        startTime: Date.now()
      };
      
      addTask(newTask);
      selectTask(id);
      emit('taskCreated', id);
      
    } catch (err) {
      console.error("Failed to start task:", err);
      message.error("无法启动任务: " + err);
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
            <n-select 
              v-model:value="selectedModel" 
              :options="availableModels" 
              placeholder="加载中..."
              style="width: 100%"
            />
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
        </n-form>

        <n-button 
          type="primary" 
          size="large" 
          @click="startNewTask" 
          block 
          style="height: 50px; font-weight: bold; margin-top: 20px;"
          :disabled="!selectedModel"
        >
          选择图片并开始处理
        </n-button>
      </n-space>
    </n-card>
  </div>
</template>

<style scoped>
.createTaskPage {
  max-width: 800px;
  margin: 0 auto;
}
</style>
