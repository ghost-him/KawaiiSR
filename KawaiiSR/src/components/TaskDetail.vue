<script setup lang="ts">
import { h, computed } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { 
  NTag, NProgress, NCard, NImage,
  NDescriptions, NDescriptionsItem, NText, useMessage, NButton
} from "naive-ui";
import type { ImageRenderToolbarProps } from 'naive-ui'
import type { Task } from "../types";
import { useTasks } from "../composables/useTasks";
import ImageCompare from "./ImageCompare.vue";

const props = defineProps<{
  task: Task
}>();

const { cancelTask } = useTasks();
const message = useMessage();

function formatStatus(status: string) {
    switch (status) {
        case 'completed': return '已完成';
        case 'processing': return '处理中';
        case 'failed': return '失败';
        case 'cancelled': return '已取消';
        default: return '未知';
    }
}

function statusType(status: string) {
    switch (status) {
        case 'completed': return 'success';
        case 'processing': return 'info';
        case 'failed': return 'error';
        case 'cancelled': return 'warning';
        default: return 'default';
    }
}

const progressText = computed(() => {
  const t = props.task;
  if (t.status === 'completed') return '转换已完成，结果已就绪';
  if (t.status === 'failed') return '处理失败';
  if (t.status === 'cancelled') return '任务已取消';
  
  if (t.status === 'processing') {
    if (t.completedTiles !== undefined && t.totalTiles !== undefined) {
      const elapsed = (Date.now() - t.startTime) / 1000;
      let etaText = "";
      if (t.completedTiles > 0) {
        const eta = (elapsed / t.completedTiles) * (t.totalTiles - t.completedTiles);
        etaText = ` | 预计剩余时间: ${Math.ceil(eta)}s`;
      }
      return `正在处理分块: ${t.completedTiles}/${t.totalTiles} (已用时: ${Math.ceil(elapsed)}s${etaText})`;
    }
    return '正在进行超分辨率处理...';
  }
  return '准备中';
});

async function saveResult(e?: MouseEvent) {
  if (e) {
    e.stopPropagation();
  }
  const taskId = props.task.id;
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
</script>

<template>
  <div class="task-detail">
    <div class="header">
      <h1 style="display: flex; align-items: center; gap: 16px; margin-bottom: 24px; margin-top: 0; font-size: 24px;">
        任务详情 (ID: {{ task.id }})
        <n-tag :type="statusType(task.status)" round :bordered="false">
          {{ formatStatus(task.status) }}
        </n-tag>
        <n-button 
          v-if="task.status === 'pending' || task.status === 'processing'" 
          size="small" 
          type="error" 
          secondary 
          round
          @click="cancelTask(task.id)"
        >
          取消任务
        </n-button>
      </h1>
    </div>
    
    <div class="info-grid">
      <n-card :bordered="false" class="info-card">
        <n-descriptions label-placement="left" :column="2" label-style="min-width: 100px; color: #666;">
          <n-descriptions-item label="输入路径">
            <n-text style="font-family: v-mono, monospace; word-break: break-all;">{{ task.inputPath }}</n-text>
          </n-descriptions-item>
          <n-descriptions-item label="倍数">
            {{ task.scaleFactor }}x
          </n-descriptions-item>
        </n-descriptions>
      </n-card>
      
      <div class="progress-section">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <h4 style="margin: 0;">处理进度</h4>
          <div style="color: #666; font-size: 13px;">
            {{ progressText }}
          </div>
        </div>
        <n-progress 
          type="line" 
          :percentage="task.progress" 
          :status="task.status === 'failed' ? 'error' : (task.status === 'completed' ? 'success' : 'default')"
          :height="10"
          processing
          indicator-placement="inside"
          :color="task.status === 'completed' ? '#18a058' : '#2080f0'"
        />
      </div>
    </div>
    
    <div class="result-section" v-if="task.originalImageSrc || task.resultImageSrc">
      <n-card 
        :bordered="false" 
        :title="task.resultImageSrc ? '对比预览 (中间滑块可拖动)' : '原图预览'" 
        size="small" 
        class="result-card"
        content-style="display: flex; flex-direction: column; overflow: hidden; height: 100%;"
      >
        <div class="image-wrapper">
          <ImageCompare 
            v-if="task.resultImageSrc && task.originalImageSrc"
            :before="task.originalImageSrc" 
            :after="task.resultImageSrc"
          />
          <n-image 
            v-else-if="task.originalImageSrc"
            :src="task.originalImageSrc" 
            class="pixelated-img"
            style="width: 100%; height: 100%;" 
            object-fit="contain" 
            :render-toolbar="renderToolbar"
          />
        </div>
        <div v-if="task.resultImageSrc" style="margin-top: 12px; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0;">
          <div style="color: #999; font-size: 12px;">
            左侧: 原图 | 右侧: 超分结果
          </div>
          <n-button size="medium" type="primary" secondary @click="saveResult()">
            下载完整结果
          </n-button>
        </div>
      </n-card>
    </div>
  </div>
</template>

<style scoped>
.task-detail {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 20px;
}

.info-grid {
  display: flex;
  flex-direction: column;
  gap: 16px;
  flex-shrink: 0;
}

.info-card {
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.result-section {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.result-card {
  flex: 1;
  display: flex;
  flex-direction: column;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  min-height: 0;
}

:deep(.result-card > .n-card__content) {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 12px !important;
}

.image-wrapper {
  flex: 1;
  background: #f0f0f0;
  text-align: center;
  border-radius: 4px;
  overflow: hidden;
  position: relative;
  min-height: 0;
}

.pixelated-img {
  width: 100%;
  height: 100%;
}

.pixelated-img :deep(img) {
  width: 100%;
  height: 100%;
  object-fit: contain;
  image-rendering: -moz-crisp-edges !important;
  image-rendering: -webkit-optimize-contrast !important;
  image-rendering: pixelated !important;
  image-rendering: crisp-edges !important;
  -ms-interpolation-mode: nearest-neighbor !important;
}
</style>
