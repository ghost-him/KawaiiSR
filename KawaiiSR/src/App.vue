<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

const inputPath = ref("");
const scaleFactor = ref(2);
const statusMsg = ref("Ready");
const taskID = ref<number | null>(null);
const resultImageSrc = ref<string | null>(null);
const isProcessing = ref(false);

let unlisten: UnlistenFn | null = null;

onMounted(async () => {
  unlisten = await listen<number>("sr-task-completed", async (event) => {
    const completedTaskId = event.payload;
    if (completedTaskId === taskID.value) {
      statusMsg.value = `Task ${completedTaskId} completed. Fetching image...`;
      await fetchResultImage(completedTaskId);
      isProcessing.value = false;
    }
  });
});

onUnmounted(() => {
  if (unlisten) unlisten();
});

async function selectFile() {
  const selected = await open({
    multiple: false,
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }]
  });
  if (selected && !Array.isArray(selected)) {
    inputPath.value = selected;
  }
}

async function startSR() {
  if (!inputPath.value) {
    statusMsg.value = "Please select an input image.";
    return;
  }

  try {
    isProcessing.value = true;
    statusMsg.value = "Starting super-resolution...";
    resultImageSrc.value = null;
    
    const id = await invoke<number>("run_super_resolution", {
      inputPath: inputPath.value,
      scaleFactor: scaleFactor.value
    });
    
    taskID.value = id;
    statusMsg.value = `Task started (ID: ${id}). Waiting for results...`;
  } catch (err) {
    statusMsg.value = `Error: ${err}`;
    isProcessing.value = false;
  }
}

async function fetchResultImage(id: number) {
  try {
    const bytes = await invoke<Uint8Array>("get_result_image", { taskId: id });
    const blob = new Blob([new Uint8Array(bytes)], { type: "image/png" });
    resultImageSrc.value = URL.createObjectURL(blob);
    statusMsg.value = "Image loaded.";
  } catch (err) {
    statusMsg.value = `Failed to fetch image: ${err}`;
  }
}

async function saveResult() {
  if (taskID.value === null) return;

  try {
    const filePath = await save({
      filters: [{ name: "PNG Image", extensions: ["png"] }],
      defaultPath: `output_${taskID.value}.png`
    });

    if (filePath) {
      await invoke("save_result_image", {
        taskId: taskID.value,
        outputPath: filePath
      });
      statusMsg.value = `Saved to ${filePath}`;
    }
  } catch (err) {
    statusMsg.value = `Save failed: ${err}`;
  }
}
</script>

<template>
  <main class="container">
    <h1>KawaiiSR - 超分辨率工具</h1>

    <div class="card">
      <div class="row">
        <input v-model="inputPath" placeholder="选择或输入图片路径..." readonly @click="selectFile" />
        <button @click="selectFile">浏览</button>
      </div>

      <div class="row" style="margin-top: 10px;">
        <label>放大倍数: </label>
        <select v-model.number="scaleFactor">
          <option :value="2">2x</option>
          <option :value="4">4x</option>
        </select>
        <button @click="startSR" :disabled="isProcessing" style="margin-left: 10px;">
          {{ isProcessing ? "运行中..." : "开始转换" }}
        </button>
      </div>
    </div>

    <div class="status-bar">
      {{ statusMsg }}
    </div>

    <div v-if="resultImageSrc" class="result-container">
      <h2>处理结果:</h2>
      <img :src="resultImageSrc" alt="Super-resolution result" class="preview-img" />
      <div class="actions">
        <button @click="saveResult">保存图片</button>
      </div>
    </div>
  </main>
</template>

<style scoped>
.container {
  max-width: 800px;
  margin: 0 auto;
}

.card {
  padding: 20px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  margin-bottom: 20px;
}

.row {
  display: flex;
  align-items: center;
  gap: 10px;
}

input {
  flex: 1;
}

.status-bar {
  padding: 10px;
  background: #eee;
  border-radius: 4px;
  font-size: 0.9em;
}

.result-container {
  margin-top: 20px;
  text-align: center;
}

.preview-img {
  max-width: 100%;
  max-height: 500px;
  border: 1px solid #ddd;
  border-radius: 8px;
}

.actions {
  margin-top: 10px;
}

@media (prefers-color-scheme: dark) {
  .card {
    background: #333;
  }
  .status-bar {
    background: #444;
  }
}
</style>
<style>
:root {
  font-family: Inter, Avenir, Helvetica, Arial, sans-serif;
  font-size: 16px;
  line-height: 24px;
  font-weight: 400;

  color: #0f0f0f;
  background-color: #f6f6f6;

  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  -webkit-text-size-adjust: 100%;
}

.container {
  margin: 0;
  padding-top: 10vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: center;
}

.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: 0.75s;
}

.logo.tauri:hover {
  filter: drop-shadow(0 0 2em #24c8db);
}

.row {
  display: flex;
  justify-content: center;
}

a {
  font-weight: 500;
  color: #646cff;
  text-decoration: inherit;
}

a:hover {
  color: #535bf2;
}

h1 {
  text-align: center;
}

input,
button {
  border-radius: 8px;
  border: 1px solid transparent;
  padding: 0.6em 1.2em;
  font-size: 1em;
  font-weight: 500;
  font-family: inherit;
  color: #0f0f0f;
  background-color: #ffffff;
  transition: border-color 0.25s;
  box-shadow: 0 2px 2px rgba(0, 0, 0, 0.2);
}

button {
  cursor: pointer;
}

button:hover {
  border-color: #396cd8;
}
button:active {
  border-color: #396cd8;
  background-color: #e8e8e8;
}

input,
button {
  outline: none;
}

#greet-input {
  margin-right: 5px;
}

@media (prefers-color-scheme: dark) {
  :root {
    color: #f6f6f6;
    background-color: #2f2f2f;
  }

  a:hover {
    color: #24c8db;
  }

  input,
  button {
    color: #ffffff;
    background-color: #0f0f0f98;
  }
  button:active {
    background-color: #0f0f0f69;
  }
}

</style>