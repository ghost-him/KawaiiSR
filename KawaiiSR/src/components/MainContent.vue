<script setup lang="ts">
import { onMounted, onUnmounted, h } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { useTasks } from "../composables/useTasks";
import type { Task } from "../types";
import { 
  NLayout, NLayoutSider, NLayoutContent,
  NButton, NList, NListItem, NThing, NTag, NProgress, NCard, NSpace, NImage,
  NEmpty, NDescriptions, NDescriptionsItem, NText, NTooltip, useMessage
} from "naive-ui";
import type { ImageRenderToolbarProps } from 'naive-ui'

const message = useMessage();
const { tasks, activeTask, addTask, updateTask, selectTask } = useTasks();

let unlisten: UnlistenFn | null = null;

onMounted(async () => {
  unlisten = await listen<number>("sr-task-completed", async (event) => {
    const completedTaskId = event.payload;
    console.log("Task completed:", completedTaskId);
    updateTask(completedTaskId, { status: "completed", progress: 100 });
    await fetchResultImage(completedTaskId);
    message.success(`任务 ${completedTaskId} 已完成`);
  });
});

onUnmounted(() => {
  if (unlisten) unlisten();
});

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
      
      const newTask: Task = {
        id,
        inputPath,
        filename,
        scaleFactor: scale,
        status: 'processing',
        progress: 0, 
        startTime: Date.now()
      };
      
      addTask(newTask);
      selectTask(id);
      simulateProgress(id);
      
    } catch (err) {
      console.error("Failed to start task:", err);
      message.error("无法启动任务: " + err);
    }
  }
}

function simulateProgress(id: number) {
  let p = 0;
  const interval = setInterval(() => {
    const task = tasks.value.find(t => t.id === id);
    if (!task || task.status !== 'processing') {
      clearInterval(interval);
      return;
    }
    if (p < 90) {
      p += Math.random() * 5;
    } else {
      p += Math.random() * 0.5;
    }
    if (p > 99) p = 99;
    updateTask(id, { progress: Math.floor(p) });
  }, 500);
}

async function fetchResultImage(id: number) {
  try {
    const bytes = await invoke<Uint8Array>("get_result_image", { taskId: id });
    const blob = new Blob([new Uint8Array(bytes)], { type: "image/png" });
    const url = URL.createObjectURL(blob);
    updateTask(id, { resultImageSrc: url });
  } catch (err) {
    console.error(`Failed to fetch image for task ${id}:`, err);
    updateTask(id, { status: 'failed' });
  }
}

async function saveResult(e?: MouseEvent) {
      if (e) {
        e.stopPropagation();
      }
      if (!activeTask.value) return;
      const taskId = activeTask.value.id;
      console.log("Saving result for task:", taskId);

      try {
        const filePath = await save({
          title: '选择保存位置',
          filters: [{ name: "PNG Image", extensions: ["png"] }],
          defaultPath: `output_${taskId}.png`
        });

        if (filePath) {
          await invoke("save_result_image", {
            taskId: taskId,
            outputPath: filePath
          });
          message.success(`已保存至 ${filePath}`);
        }
      } catch (err) {
        console.error("Save error:", err);
    message.error(`保存失败: ${err}`);
  }
}

function renderToolbar({ nodes }: ImageRenderToolbarProps) {
    return Object.entries(nodes).map(([key, node]) => {
        if (key === "download") {
            // Completely recreate the VNode to replace the onClick handler instead of merging it
            return h(
                node.type as any,
                {
                    ...node.props,
                    onClick: (e: MouseEvent) => {
                        e.stopPropagation();
                        e.preventDefault();
                        saveResult(e);
                    }
                },
                node.children as any
            )
        }
        return node
    });
}

function formatStatus(status: string) {
    switch (status) {
        case 'completed': return '已完成';
        case 'processing': return '处理中';
        case 'failed': return '失败';
        default: return '未知';
    }
}
function statusType(status: string) {
    switch (status) {
        case 'completed': return 'success';
        case 'processing': return 'info';
        case 'failed': return 'error';
        default: return 'default';
    }
}
</script>

<template>
  <n-layout has-sider style="height: 100vh">
    <n-layout-sider
      bordered
      width="320"
      content-style="padding: 24px; display: flex; flex-direction: column; background-color: #fff;"
    >
      <div style="margin-bottom: 24px;">
        <n-button type="info" block @click="startNewTask" size="large" style="height: 50px; font-weight: bold; font-size: 16px;">
          + 新建任务
        </n-button>
      </div>
      
      <n-list style="background: transparent;">
        <n-list-item 
          v-for="task in tasks" 
          :key="task.id"
          @click="selectTask(task.id)"
          style="cursor: pointer; padding: 12px; border-radius: 8px; margin-bottom: 8px; transition: all 0.2s;"
          :class="{ 'selected-task': activeTask?.id === task.id }"
        >
          <n-thing content-style="margin-top: 0;">
            <template #header>
              <n-tooltip trigger="hover" placement="top-start">
                <template #trigger>
                  <div style="font-weight: bold; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 220px;">
                    {{ task.filename }}
                  </div>
                </template>
                {{ task.filename }}
              </n-tooltip>
            </template>
            <template #description>
              <n-space justify="space-between" align="center" style="margin-top: 8px;">
                <n-tag :type="statusType(task.status)" size="small" :bordered="false" round>
                  {{ formatStatus(task.status) }}
                </n-tag>
              </n-space>
            </template>
          </n-thing>
        </n-list-item>
        <div v-if="tasks.length === 0" style="text-align: center; margin-top: 40px; color: #999;">
          暂无任务，请新建
        </div>
      </n-list>
    </n-layout-sider>
    
    <n-layout-content content-style="padding: 48px; background-color: #f9f9f9;">
      <div v-if="activeTask" class="task-detail">
        <div class="header">
          <h1 style="display: flex; align-items: center; gap: 16px; margin-bottom: 40px; margin-top: 0; font-size: 24px;">
            任务详情 (ID: {{ activeTask.id }})
            <n-tag :type="statusType(activeTask.status)" round :bordered="false">
              {{ formatStatus(activeTask.status) }}
            </n-tag>
          </h1>
        </div>
        
        <n-card :bordered="false" style="margin-bottom: 30px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
          <n-descriptions label-placement="left" :column="1" label-style="min-width: 100px; color: #666;">
            <n-descriptions-item label="输入路径">
              <n-text style="font-family: v-mono, monospace; word-break: break-all;">{{ activeTask.inputPath }}</n-text>
            </n-descriptions-item>
            <n-descriptions-item label="已设定倍数">
              {{ activeTask.scaleFactor }}x
            </n-descriptions-item>
          </n-descriptions>
        </n-card>
        
        <div class="progress-section" style="margin-bottom: 48px;">
          <h4 style="margin-bottom: 16px; margin-top: 0;">处理进度</h4>
          <n-progress 
            type="line" 
            :percentage="activeTask.progress" 
            :status="activeTask.status === 'failed' ? 'error' : (activeTask.status === 'completed' ? 'success' : 'default')"
            :height="12"
            processing
            indicator-placement="inside"
            :color="activeTask.status === 'completed' ? '#18a058' : '#2080f0'"
          />
          <div style="margin-top: 12px; color: #666; font-size: 14px;">
            {{ activeTask.status === 'completed' ? '转换已完成，结果已就绪' : (activeTask.status === 'processing' ? '正在进行超分辨率处理...' : '准备中') }}
          </div>
        </div>
        
        <div class="result-section" v-if="activeTask.resultImageSrc">
          <n-card :bordered="false" title="处理结果预览" size="small" style="border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.06);">
            <div class="image-wrapper" style="background: white; padding: 12px; text-align: center; border-radius: 4px; overflow: hidden;">
              <n-image 
                :src="activeTask.resultImageSrc" 
                class="pixelated-img"
                style="max-width: 100%; max-height: 500px;" 
                object-fit="contain" 
                :render-toolbar="renderToolbar"
              />
            </div>
            <div style="margin-top: 12px; text-align: center; color: #999; font-size: 12px;">
              点击图片可查看大图并下载
            </div>
          </n-card>
        </div>
      </div>
      <div v-else class="empty-state" style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;">
        <n-empty description="请从左侧选择任务或新建任务" size="large">
          <template #icon>
            <div style="font-size: 64px;">📦</div>
          </template>
        </n-empty>
      </div>
    </n-layout-content>
  </n-layout>
</template>

<style scoped>
.selected-task {
  background-color: #eaf2ff;
  border: 1px solid #d0e1ff;
}
.selected-task :deep(.n-thing-header__title) {
  color: #2080f0;
}
.pixelated-img :deep(img) {
  image-rendering: -moz-crisp-edges !important;
  image-rendering: -webkit-optimize-contrast !important;
  image-rendering: pixelated !important;
  image-rendering: crisp-edges !important;
  -ms-interpolation-mode: nearest-neighbor !important;
}
</style>

<style>
/* Global style to target Naive UI Image Preview */
.n-image-preview-container img,
.n-image-preview-img {
  image-rendering: -moz-crisp-edges !important;
  image-rendering: -webkit-optimize-contrast !important;
  image-rendering: pixelated !important;
  image-rendering: crisp-edges !important;
  -ms-interpolation-mode: nearest-neighbor !important;
}
</style>
