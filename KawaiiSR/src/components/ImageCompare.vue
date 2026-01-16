<script setup lang="ts">
import { ref, computed, watch } from 'vue';

const props = defineProps<{
  before: string;
  after: string;
}>();

const containerRef = ref<HTMLElement | null>(null);
const sliderPos = ref(50);
const scale = ref(1);
const translateX = ref(0);
const translateY = ref(0);

const isPanning = ref(false);
const isSliding = ref(false);
const startX = ref(0);
const startY = ref(0);
const containerWidth = ref(0);

// Update container width on mount and resize
const updateContainerWidth = () => {
  if (containerRef.value) {
    containerWidth.value = containerRef.value.getBoundingClientRect().width;
  }
};

// Reset zoom when images change
watch(() => props.after, () => {
  scale.value = 1;
  translateX.value = 0;
  translateY.value = 0;
  updateContainerWidth();
});

import { onMounted, onUnmounted } from 'vue';
onMounted(() => {
  updateContainerWidth();
  window.addEventListener('resize', updateContainerWidth);
});
onUnmounted(() => {
  window.removeEventListener('resize', updateContainerWidth);
});

function handleMouseDown(e: MouseEvent) {
  if (e.button === 0) { // Left click
    isSliding.value = true;
    updateSlider(e);
  } else if (e.button === 2) { // Right click
    isPanning.value = true;
    startX.value = e.clientX - translateX.value;
    startY.value = e.clientY - translateY.value;
  }
}

function handleMouseMove(e: MouseEvent) {
  if (isSliding.value) {
    updateSlider(e);
  } else if (isPanning.value) {
    translateX.value = e.clientX - startX.value;
    translateY.value = e.clientY - startY.value;
  }
}

function handleMouseUp() {
  isSliding.value = false;
  isPanning.value = false;
}

function updateSlider(e: MouseEvent) {
  if (!containerRef.value) return;
  const rect = containerRef.value.getBoundingClientRect();
  const screenX = e.clientX - rect.left;
  // Convert screen coordinates to local coordinates within the transformed layer
  const localX = (screenX - translateX.value) / scale.value;
  sliderPos.value = Math.max(0, Math.min(100, (localX / rect.width) * 100));
}

function handleWheel(e: WheelEvent) {
  e.preventDefault();
  if (!containerRef.value) return;

  const rect = containerRef.value.getBoundingClientRect();
  containerWidth.value = rect.width; // Keep width up to date
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  // Faster zoom when bigger, slower when smaller
  const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
  const nextScale = Math.max(0.1, Math.min(scale.value * zoomFactor, 50));

  // Zoom towards mouse
  const zoomRatio = nextScale / scale.value;
  translateX.value = mouseX - (mouseX - translateX.value) * zoomRatio;
  translateY.value = mouseY - (mouseY - translateY.value) * zoomRatio;
  scale.value = nextScale;
}

function resetView() {
  scale.value = 1;
  translateX.value = 0;
  translateY.value = 0;
}

const transformStyle = computed(() => ({
  transform: `translate(${translateX.value}px, ${translateY.value}px) scale(${scale.value})`,
  transformOrigin: '0 0'
}));

const sliderLineStyle = computed(() => {
  // Use the formula: screenX = translateX + (sliderPos/100 * containerWidth * scale)
  const left = translateX.value + (sliderPos.value / 100) * containerWidth.value * scale.value;
  return {
    left: `${left}px`
  };
});

</script>

<template>
  <div 
    class="image-compare-container" 
    ref="containerRef"
    @mousedown="handleMouseDown"
    @mousemove="handleMouseMove"
    @mouseup="handleMouseUp"
    @mouseleave="handleMouseUp"
    @wheel="handleWheel"
    @contextmenu.prevent
  >
    <div class="transform-layer" :style="transformStyle">
      <img :src="after" class="image-after" draggable="false" />
      <img 
        :src="before" 
        class="image-before" 
        :style="{ 
          clipPath: `inset(0 ${100 - sliderPos}% 0 0)`
        }"
        draggable="false"
      />
    </div>
    
    <div 
      class="slider-line" 
      :style="sliderLineStyle"
    >
      <div class="slider-button">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="15 18 9 12 15 6"></polyline>
          <polyline points="9 18 3 12 9 6"></polyline>
        </svg>
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="9 18 15 12 9 6"></polyline>
          <polyline points="15 18 21 12 15 6"></polyline>
        </svg>
      </div>
    </div>
    
    <div class="controls-overlay">
      <div class="zoom-tag" v-if="scale !== 1">
        {{ Math.round(scale * 100) }}%
        <div class="reset-btn" @click.stop="resetView">重置</div>
      </div>
      <div class="hint-tag">左键滑动 | 右键拖拽 | 滚轮缩放</div>
    </div>
  </div>
</template>

<style scoped>
.image-compare-container {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 400px;
  overflow: hidden;
  border-radius: 8px;
  background-color: #f0f0f0;
  user-select: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: crosshair;
}

.transform-layer {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  /* Removing will-change as it can sometimes cause browsers to cache a low-res 
     snapshot of the layer, leading to blurry results when scaling up. */
  image-rendering: -webkit-optimize-contrast;
  image-rendering: -moz-crisp-edges;
  image-rendering: crisp-edges;
  image-rendering: pixelated;
  transform-style: preserve-3d;
}

.image-before,
.image-after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  /* Multiple rendering hints to force pixelated/sharp edges across all engines */
  image-rendering: -webkit-optimize-contrast;
  image-rendering: -moz-crisp-edges;
  image-rendering: crisp-edges;
  image-rendering: pixelated;
  -ms-interpolation-mode: nearest-neighbor;
}

.image-before {
  z-index: 2;
}

.image-after {
  z-index: 1;
}

.slider-line {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background-color: white;
  z-index: 3;
  pointer-events: none;
  box-shadow: 0 0 4px rgba(0,0,0,0.5);
  transform: translateX(-1px);
}

.slider-button {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  color: #666;
}

.slider-button svg {
  width: 12px;
  height: 12px;
  margin: 0 -1px;
}

.controls-overlay {
  position: absolute;
  bottom: 12px;
  left: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 10;
  pointer-events: none;
}

.zoom-tag, .hint-tag {
  background: rgba(0, 0, 0, 0.6);
  color: white;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 11px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.hint-tag {
  opacity: 0.7;
}

.reset-btn {
  background: #2080f0;
  padding: 2px 6px;
  border-radius: 3px;
  cursor: pointer;
  pointer-events: auto;
}

.reset-btn:hover {
  background: #4098fc;
}
</style>
