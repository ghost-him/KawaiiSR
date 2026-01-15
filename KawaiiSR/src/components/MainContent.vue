<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { useTasks } from "../composables/useTasks";
import { 
  NLayout, NLayoutSider, NLayoutContent,
  NEmpty, useMessage
} from "naive-ui";

// Components
import TaskList from "./TaskList.vue";
import TaskDetail from "./TaskDetail.vue";
import CreateTask from "./CreateTask.vue";

const message = useMessage();
const { activeTask, updateTask, selectTask, fetchResultImage } = useTasks();

const isCreating = ref(false);

interface ProgressPayload {
  task_id: number;
  completed_tiles: number;
  total_tiles: number;
}

let unlisten: UnlistenFn | null = null;
let unlistenProgress: UnlistenFn | null = null;

onMounted(async () => {
  unlisten = await listen<number>("sr-task-completed", async (event) => {
    const completedTaskId = event.payload;
    console.log("Task completed:", completedTaskId);
    updateTask(completedTaskId, { status: "completed", progress: 100 });
    await fetchResultImage(completedTaskId);
    message.success(`任务 ${completedTaskId} 已完成`);
  });

  unlistenProgress = await listen<ProgressPayload>("sr-task-progress", (event) => {
    const { task_id, completed_tiles, total_tiles } = event.payload;
    const progress = Math.floor((completed_tiles / total_tiles) * 100);
    updateTask(task_id, { 
      progress, 
      completedTiles: completed_tiles, 
      totalTiles: total_tiles 
    });
  });
});

onUnmounted(() => {
  if (unlisten) unlisten();
  if (unlistenProgress) unlistenProgress();
});

function handleNewTaskClick() {
  isCreating.value = true;
  selectTask(null);
}

function handleSelectTask(id: number) {
  isCreating.value = false;
  selectTask(id);
}

function onTaskCreated() {
  isCreating.value = false;
}
</script>

<template>
  <n-layout has-sider style="height: 100vh">
    <n-layout-sider
      bordered
      width="320"
      content-style="padding: 24px; display: flex; flex-direction: column; background-color: #fff;"
    >
      <TaskList 
        :is-creating="isCreating" 
        @add-new-task="handleNewTaskClick" 
        @select-task="handleSelectTask" 
      />
    </n-layout-sider>
    
    <n-layout-content content-style="padding: 48px; background-color: #f9f9f9;">
      <CreateTask 
        v-if="isCreating" 
        @task-created="onTaskCreated" 
      />

      <TaskDetail 
        v-else-if="activeTask" 
        :task="activeTask" 
      />

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
