<script setup lang="ts">
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { 
  NButton, NCard, NSpace, NText, useMessage
} from "naive-ui";
import { useTasks } from "../composables/useTasks";
import type { Task } from "../types";

const emit = defineEmits(['taskCreated']);
const message = useMessage();
const { addTask, selectTask } = useTasks();

interface TaskMetadata {
  total_tiles: number;
}

async function startNewTask() {
  const selected = await open({
    multiple: false,
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }]
  });
  
  if (selected && !Array.isArray(selected)) {
    const inputPath = selected;
    const filename = inputPath.split(/[\\/]/).pop() || "image.png";
    const scale = 2; 
    
    try {
      const id = await invoke<number>("run_super_resolution", {
        inputPath: inputPath,
        scaleFactor: scale
      });
      
      const metadata = await invoke<TaskMetadata>("get_task_metadata", { taskId: id });
      
      const newTask: Task = {
        id,
        inputPath,
        filename,
        scaleFactor: scale,
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
          配置您的超分辨率参数。目前暂无更多选项，点击下方按钮选择图片并开始处理。
        </n-text>
        
        <div style="padding: 40px; border: 2px dashed #e0e0e0; border-radius: 12px; text-align: center; color: #999; background: #fafafa;">
          <n-text depth="3" style="font-size: 16px;">
            更多选项（如模型选择、缩放倍数等）即将推出...
          </n-text>
        </div>

        <n-button type="primary" size="large" @click="startNewTask" block style="height: 50px; font-weight: bold;">
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
