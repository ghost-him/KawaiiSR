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
      <h1 style="display: flex; align-items: center; gap: 16px; margin-bottom: 40px; margin-top: 0; font-size: 24px;">
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
    
    <n-card :bordered="false" style="margin-bottom: 30px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
      <n-descriptions label-placement="left" :column="1" label-style="min-width: 100px; color: #666;">
        <n-descriptions-item label="输入路径">
          <n-text style="font-family: v-mono, monospace; word-break: break-all;">{{ task.inputPath }}</n-text>
        </n-descriptions-item>
        <n-descriptions-item label="已设定倍数">
          {{ task.scaleFactor }}x
        </n-descriptions-item>
      </n-descriptions>
    </n-card>
    
    <div class="progress-section" style="margin-bottom: 48px;">
      <h4 style="margin-bottom: 16px; margin-top: 0;">处理进度</h4>
      <n-progress 
        type="line" 
        :percentage="task.progress" 
        :status="task.status === 'failed' ? 'error' : (task.status === 'completed' ? 'success' : 'default')"
        :height="12"
        processing
        indicator-placement="inside"
        :color="task.status === 'completed' ? '#18a058' : '#2080f0'"
      />
      <div style="margin-top: 12px; color: #666; font-size: 14px;">
        {{ progressText }}
      </div>
    </div>
    
    <div class="result-section" v-if="task.resultImageSrc">
      <n-card :bordered="false" title="处理结果预览" size="small" style="border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.06);">
        <div class="image-wrapper" style="background: white; padding: 12px; text-align: center; border-radius: 4px; overflow: hidden;">
          <n-image 
            :src="task.resultImageSrc" 
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
</template>

<style scoped>
.pixelated-img :deep(img) {
  image-rendering: -moz-crisp-edges !important;
  image-rendering: -webkit-optimize-contrast !important;
  image-rendering: pixelated !important;
  image-rendering: crisp-edges !important;
  -ms-interpolation-mode: nearest-neighbor !important;
}
</style>
