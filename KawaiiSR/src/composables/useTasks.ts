import { ref, computed } from 'vue';
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

    function selectTask(id: number) {
        activeTaskId.value = id;
    }

    return {
        tasks,
        activeTaskId,
        activeTask,
        addTask,
        updateTask,
        selectTask
    };
}
