export interface Task {
    id: number;
    inputPath: string;
    outputPath?: string;
    filename: string;
    scaleFactor: number;
    modelName: string;
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    completedTiles?: number;
    totalTiles?: number;
    resultImageSrc?: string;
    originalImageSrc?: string;
    startTime: number;
}
