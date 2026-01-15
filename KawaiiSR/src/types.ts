export interface Task {
    id: number;
    inputPath: string;
    outputPath?: string;
    filename: string;
    scaleFactor: number;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress: number;
    completedTiles?: number;
    totalTiles?: number;
    resultImageSrc?: string;
    startTime: number;
}
