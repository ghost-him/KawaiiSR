# 开发者指南

如果要在本地运行该项目，执行以下的命令即可。当前只维护了 Windows 11 版本：

1. 克隆仓库
   ```sh
   git clone https://github.com/ghost-him/KawaiiSR.git
   ```
2. 安装前端依赖
   ```sh
   cd KawaiiSR/KawaiiSR
   bun install
   ```
3. 下载模型权重到 src-tauri/onnx 文件夹下（如果没有curl，可以手动下载模型）
   ```sh
   curl -f -L -v -o KawaiiSR/src-tauri/onnx/KawaiiSR-precise.onnx https://github.com/ghost-him/KawaiiSR/releases/download/model/KawaiiSR-precise.onnx
   ```
 
5. 启动开发模式
   ```sh
   bun run tauri dev
   ```
6. 打包
   ```sh
   bun run tauri build
   ```
