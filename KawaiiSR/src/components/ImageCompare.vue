<script setup lang="ts">
import { ref } from 'vue';

const props = defineProps<{
  before: string;
  after: string;
}>();

const sliderPos = ref(50);

function handleInput(e: Event) {
  const target = e.target as HTMLInputElement;
  sliderPos.value = Number(target.value);
}

</script>

<template>
  <div class="image-compare-container">
    <img :src="after" class="image-after" />
    <img 
      :src="before" 
      class="image-before" 
      :style="{ 
        clipPath: `inset(0 ${100 - sliderPos}% 0 0)`
      }"
    />
    
    <div class="slider-line" :style="{ left: `${sliderPos}%` }">
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
    
    <input 
      type="range" 
      min="0" 
      max="100" 
      :value="sliderPos" 
      @input="handleInput"
      class="slider-input"
    />
  </div>
</template>

<style scoped>
.image-compare-container {
  position: relative;
  width: 100%;
  height: 500px;
  overflow: hidden;
  border-radius: 8px;
  background-color: #f0f0f0;
  user-select: none;
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-before,
.image-after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  /* Ensure images match precisely */
  display: block;
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
  image-rendering: pixelated;
}

.image-before {
  z-index: 2;
  image-rendering: pixelated; /* Show original pixels if zoomed */
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

.slider-input {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  z-index: 4;
  cursor: ew-resize;
}

/* Optional: style the range input for better interaction if needed, 
   but opacity 0 hides it while keeping functionality */
</style>
