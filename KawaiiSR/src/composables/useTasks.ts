import { ref, computed } from 'vue';
import { invoke } from "@tauri-apps/api/core";
import type { Task } from '../types';

const tasks = ref<Task[]>([]);
const activeTaskId = ref<number | null>(null);

export function useTasks() {
    const activeTask = computed(() =>
        tasks.value.find(t => t.id === activeTaskId.value) || null
    );

    function addTask(task: Task, setActive: boolean = true) {
        tasks.value.unshift(task);
        if (setActive) {
            activeTaskId.value = task.id;
        }
        if (task.inputPath) {
            fetchOriginalImage(task.id, task.inputPath);
        }
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

    async function fetchOriginalImage(id: number, path: string) {
        try {
            const bytes = await invoke<Uint8Array>("get_image_data", { path });
            const blob = new Blob([new Uint8Array(bytes)], { type: "image/png" });
            const url = URL.createObjectURL(blob);
            updateTask(id, { originalImageSrc: url });
        } catch (err) {
            console.error(`Failed to fetch original image for task ${id}:`, err);
        }
    }

    async function fetchResultImage(id: number) {
        try {
            const bytes = await invoke<number[]>("get_result_image", { taskId: id });
            const uint8Bytes = new Uint8Array(bytes);
            const blob = new Blob([uint8Bytes], { type: "image/png" });
            const url = URL.createObjectURL(blob);
            updateTask(id, { resultImageSrc: url, outputSize: uint8Bytes.length });
        } catch (err) {
            console.error(`Failed to fetch image for task ${id}:`, err);
            updateTask(id, { status: 'failed' });
        }
    }

    async function cancelTask(id: number) {
        try {
            // 首先更新本地状态，让用户感觉到反馈
            updateTask(id, { status: 'cancelled' });
            await invoke("cancel_super_resolution", { taskId: id });
        } catch (err) {
            console.error(`Failed to cancel task ${id}:`, err);
        }
    }

    async function getAvailableModels(): Promise<string[]> {
        try {
            return await invoke<string[]>("get_available_models");
        } catch (err) {
            console.error("Failed to fetch available models:", err);
            return [];
        }
    }

    async function getDefaultModel(): Promise<string> {
        try {
            return await invoke<string>("get_default_model");
        } catch (err) {
            console.error("Failed to fetch default model:", err);
            return "";
        }
    }

    return {
        tasks,
        activeTaskId,
        activeTask,
        addTask,
        updateTask,
        selectTask,
        fetchResultImage,
        cancelTask,
        getAvailableModels,
        getDefaultModel
    };
}
