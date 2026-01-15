<script setup lang="ts">
import { NConfigProvider, NMessageProvider, NGlobalStyle } from "naive-ui";
import MainContent from "./components/MainContent.vue";
import 'vfonts/Lato.css'
import 'vfonts/FiraCode.css'
</script>

<template>
  <n-config-provider>
    <n-global-style />
    <n-message-provider>
      <MainContent />
    </n-message-provider>
  </n-config-provider>
</template>

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
</style>
