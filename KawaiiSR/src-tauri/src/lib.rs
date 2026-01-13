use image::{GenericImageView, ImageBuffer, Rgb};
use ndarray::{Array3, Array4, Axis};
use ort::{
    execution_providers::DirectMLExecutionProvider,
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn run_super_resolution() -> Result<String, String> {
    println!("Command 'run_super_resolution' invoked");
    inner_run_super_resolution().map_err(|e| {
        let err_msg = format!("SR Error: {:?}", e);
        eprintln!("{}", err_msg);
        err_msg
    })
}

fn inner_run_super_resolution() -> Result<String, Box<dyn std::error::Error>> {
    // 1. Resolve paths
    let cwd = std::env::current_dir()?;
    println!("Current working directory: {:?}", cwd);

    let mut model_path = cwd.join("onnx/kawaii_sr.onnx");
    if !model_path.exists() {
        model_path = cwd.join("src-tauri/onnx/kawaii_sr.onnx");
    }

    println!("Attempting to load model from: {:?}", model_path);
    if !model_path.exists() {
        return Err(format!("Model not found at {:?}", model_path).into());
    }

    // Try loading model with execution provider fallback
    // Default to DirectML, fallback to CPU if DirectML fails
    // Note: on Windows with OpenMP, with_intra_threads() has no effect.
    // Set OMP_NUM_THREADS environment variable instead for thread control.
    let use_directml = std::env::var("USE_DIRECTML").unwrap_or_else(|_| "1".to_string()) == "1";

    let mut session = if use_directml {
        println!("Attempting to load with DirectML execution provider...");
        match Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_execution_providers([DirectMLExecutionProvider::default().build()])?
            .commit_from_file(&model_path)
        {
            Ok(s) => {
                println!("✓ Model loaded successfully with DirectML");
                s
            }
            Err(e) => {
                eprintln!("✗ DirectML failed: {:?}, falling back to CPU", e);
                Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_parallel_execution(true)?
                    .commit_from_file(&model_path)?
            }
        }
    } else {
        println!("Attempting to load with CPU execution provider (DirectML disabled)...");
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_parallel_execution(true)?
            .commit_from_file(&model_path)?
    };

    println!("Model loaded successfully");

    // 2. Load and preprocess image
    let mut img_path = cwd.join("../test.png");
    if !img_path.exists() {
        img_path = cwd.join("test.png");
    }

    println!("Attempting to load image from: {:?}", img_path);
    if !img_path.exists() {
        return Err(format!("Input image not found at {:?}. CWD is {:?}", img_path, cwd).into());
    }

    let img = image::open(&img_path)?;
    let (width, height) = img.dimensions();

    // Convert to RGB8
    let rgb_img = img.to_rgb8();

    // Prepare input tensor [1, 3, H, W] using vectorized ndarray operations
    // Fix: Use as_standard_layout().to_owned() to ensure contiguous CHW memory for the ONNX tensor
    let input_array =
        Array3::from_shape_vec((height as usize, width as usize, 3), rgb_img.into_raw())?
            .permuted_axes([2, 0, 1]) // HWC -> CHW
            .mapv(|x| x as f32 / 255.0)
            .insert_axis(Axis(0)) // CHW -> NCHW
            .as_standard_layout()
            .to_owned();

    // 3. Run inference
    let input_tensor = Value::from_array(input_array)?;
    let outputs = session.run(inputs![
        "input" => input_tensor,
    ])?;

    let output_value = outputs
        .get("output")
        .ok_or("Failed to get output 'output'")?;
    let (output_shape, output_data) = output_value.try_extract_tensor::<f32>()?;

    // 4. Post-process [1, 3, H*S, W*S]
    let out_h = output_shape[2] as usize;
    let out_w = output_shape[3] as usize;

    // Fix: Use as_standard_layout().to_owned() before into_raw_vec_and_offset()
    // to ensure RGB pixels are interleaved correctly for image::ImageBuffer
    let out_array = Array4::from_shape_vec((1, 3, out_h, out_w), output_data.to_vec())?
        .index_axis_move(Axis(0), 0) // [3, H, W]
        .permuted_axes([1, 2, 0]) // [H, W, 3]
        .as_standard_layout()
        .to_owned()
        .mapv(|x| (x * 255.0).clamp(0.0, 255.0) as u8);

    let out_img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(
        out_w as u32,
        out_h as u32,
        out_array.into_raw_vec_and_offset().0,
    )
    .ok_or("Failed to create ImageBuffer from raw output data")?;

    // 5. Save result
    let out_path = img_path.with_file_name("test_output.png");
    out_img.save(&out_path)?;

    Ok(format!(
        "Super-resolution completed. Result saved to {:?}",
        out_path
    ))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, run_super_resolution])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
