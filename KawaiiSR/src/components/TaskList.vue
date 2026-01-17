<script setup lang="ts">
import { 
  NList, NListItem, NThing, NTag, NTooltip, NButton, NSpace
} from "naive-ui";
import { useTasks } from "../composables/useTasks";

defineProps<{
  isCreating: boolean
}>();

const emit = defineEmits(['addNewTask', 'selectTask']);
const { tasks, activeTaskId } = useTasks();

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
</script>

<template>
  <div class="sidebar-container">
    <div style="margin-bottom: 24px;">
      <n-button type="info" block @click="emit('addNewTask')" size="large" style="height: 50px; font-weight: bold; font-size: 16px;">
        + 新建任务
      </n-button>
    </div>
    
    <n-list style="background: transparent;">
      <n-list-item 
        v-for="task in tasks" 
        :key="task.id"
        @click="emit('selectTask', task.id)"
        style="cursor: pointer; padding: 12px; border-radius: 8px; margin-bottom: 8px; transition: all 0.2s;"
        :class="{ 'selected-task': !isCreating && activeTaskId === task.id }"
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
            <div style="font-size: 12px; color: #888; margin-top: 4px;">
              {{ task.modelName }} | {{ task.scaleFactor }}x
            </div>
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
  </div>
</template>

<style scoped>
.sidebar-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.selected-task {
  background-color: #eaf2ff;
  border: 1px solid #d0e1ff;
}
.selected-task :deep(.n-thing-header__title) {
  color: #2080f0;
}
</style>
