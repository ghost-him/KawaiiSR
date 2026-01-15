import { ref, computed } from 'vue';
import { invoke } from "@tauri-apps/api/core";
import type { Task } from '../types';

const tasks = ref<Task[]>([]);
const activeTaskId = ref<number | null>(null);

export function useTasks() {
    const activeTask = computed(() =>
        tasks.value.find(t => t.id === activeTaskId.value) || null
    );

    function addTask(task: Task) {
        tasks.value.unshift(task);
        activeTaskId.value = task.id;
    }

    function updateTask(id: number, updates: Partial<Task>) {
        const task = tasks.value.find(t => t.id === id);
        if (task) {
            Object.assign(task, updates);
        }
    }

    function selectTask(id: number | null) {
        activeTaskId.value = id;
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

    return {
        tasks,
        activeTaskId,
        activeTask,
        addTask,
        updateTask,
        selectTask,
        fetchResultImage
    };
}
