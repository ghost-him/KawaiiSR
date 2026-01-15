import argparse
from pathlib import Path
import math
import numpy as np
from PIL import Image
import onnxruntime as ort


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return arr  # (H, W, 3)


def save_image(arr: np.ndarray, path: Path) -> None:
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def get_session(onnx_path: Path, use_cpu: bool = False) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"] if use_cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 48
    return ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)


def pad_to_window(tile: np.ndarray, window_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = tile.shape[-2:]
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return tile, (0, 0)
    # Reflect pad on H and W
    tile = np.pad(tile, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    return tile, (pad_h, pad_w)


def run_tile(session: ort.InferenceSession, tile: np.ndarray, window_size: int, scale: int) -> np.ndarray:
    tile, (pad_h, pad_w) = pad_to_window(tile, window_size)
    out = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: tile})[0]
    if pad_h or pad_w:
        h_out = (tile.shape[-2] - pad_h) * scale
        w_out = (tile.shape[-1] - pad_w) * scale
        out = out[:, :, :h_out, :w_out]
    return out


def tiled_sr(session: ort.InferenceSession, img_chw: np.ndarray, scale: int, window_size: int, tile_size: int, tile_pad: int) -> np.ndarray:
    b, c, h, w = img_chw.shape
    out = np.zeros((b, c, h * scale, w * scale), dtype=np.float32)
    tiles_x = math.ceil(w / tile_size)
    tiles_y = math.ceil(h / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            x0 = x * tile_size
            x1 = min(x0 + tile_size, w)
            y0 = y * tile_size
            y1 = min(y0 + tile_size, h)

            x0_pad = max(x0 - tile_pad, 0)
            x1_pad = min(x1 + tile_pad, w)
            y0_pad = max(y0 - tile_pad, 0)
            y1_pad = min(y1 + tile_pad, h)

            tile = img_chw[:, :, y0_pad:y1_pad, x0_pad:x1_pad]
            out_tile = run_tile(session, tile, window_size, scale)

            rel_x0 = (x0 - x0_pad) * scale
            rel_x1 = (x1 - x0_pad) * scale
            rel_y0 = (y0 - y0_pad) * scale
            rel_y1 = (y1 - y0_pad) * scale

            out_patch = out_tile[:, :, rel_y0:rel_y1, rel_x0:rel_x1]
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = out_patch
    return out


def sr_image(session: ort.InferenceSession, img: np.ndarray, scale: int, window_size: int, tile_size: int, tile_pad: int, batch_size: int = 1) -> list[np.ndarray]:
    chw = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
    if batch_size > 1:
        chw = np.tile(chw, (batch_size, 1, 1, 1))

    print(f"Running inference with input shape: {chw.shape}")
    out = tiled_sr(session, chw, scale=scale, window_size=window_size, tile_size=tile_size, tile_pad=tile_pad)
    print(f"Output shape: {out.shape}")

    # Return a list of Reconstruction images
    return [np.transpose(out[i], (1, 2, 0)) for i in range(out.shape[0])]


def parse_args():
    p = argparse.ArgumentParser(description="ONNX Super-Resolution inference for KawaiiSR")
    p.add_argument("--model", type=Path, default=Path("models/kawaii_sr.onnx"), help="Path to ONNX model")
    p.add_argument("--input", type=Path, required=True, help="Input image or folder")
    p.add_argument("--output", type=Path, default=Path("sr_outputs_onnx"), help="Output folder")
    p.add_argument("--scale", type=int, default=2, help="Upscale factor of the model")
    p.add_argument("--window-size", type=int, default=16, help="Window size used during training")
    p.add_argument("--tile-size", type=int, default=256, help="Core tile size before padding")
    p.add_argument("--tile-pad", type=int, default=48, help="Padding around each tile")
    p.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=1, help="Batch size for testing dynamic batch")
    p.add_argument("--cpu", action="store_true", help="Force CPU execution")
    p.add_argument("--exts", type=str, default="png,jpg,jpeg,webp", help="Image extensions, comma separated")
    return p.parse_args()


def collect_images(path: Path, exts: list[str]) -> list[Path]:
    if path.is_file():
        return [path]
    imgs: list[Path] = []
    for ext in exts:
        imgs.extend(path.rglob(f"*.{ext}"))
        imgs.extend(path.rglob(f"*.{ext.upper()}"))
    return sorted(set(imgs))


def main():
    args = parse_args()
    exts = [e.strip().lstrip('.') for e in args.exts.split(',') if e.strip()]
    imgs = collect_images(args.input, exts)
    if not imgs:
        raise SystemExit(f"No images found in {args.input}")

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    session = get_session(args.model, use_cpu=args.cpu)

    for img_path in imgs:
        img = load_image(img_path)
        sr_list = sr_image(
            session, img,
            scale=args.scale,
            window_size=args.window_size,
            tile_size=args.tile_size,
            tile_pad=args.tile_pad,
            batch_size=args.batch_size
        )

        for i, sr in enumerate(sr_list):
            if len(sr_list) > 1:
                save_name = f"{img_path.stem}_x{args.scale}_b{i}.png"
            else:
                save_name = f"{img_path.stem}_x{args.scale}.png"
            save_image(sr, out_dir / save_name)
            print(f"[done] {img_path.name} (batch index {i}) -> {save_name}")


if __name__ == "__main__":
    main()
