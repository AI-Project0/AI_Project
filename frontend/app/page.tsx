"use client";

import React, { useState, useCallback } from "react";
import Sidebar, { SizeId, BgColor, OutfitType } from "./components/Sidebar";
import CropperComponent from "./components/CropperComponent";
import { Area } from "react-easy-crop";
import { Upload, Download, RefreshCcw, XCircle, FileArchive, Grid, User, X, AlertCircle } from "lucide-react";
import JSZip from "jszip";
import { saveAs } from "file-saver";

// --- Helper: Get Cropped Image Blob ---
const createImage = (url: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.addEventListener("load", () => resolve(image));
    image.addEventListener("error", (error) => reject(error));
    image.setAttribute("crossOrigin", "anonymous");
    image.src = url;
  });

async function getCroppedImg(
  imageSrc: string,
  pixelCrop: Area
): Promise<Blob> {
  const image = await createImage(imageSrc);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  if (!ctx) {
    throw new Error("No 2d context");
  }

  canvas.width = pixelCrop.width;
  canvas.height = pixelCrop.height;

  ctx.drawImage(
    image,
    pixelCrop.x,
    pixelCrop.y,
    pixelCrop.width,
    pixelCrop.height,
    0,
    0,
    pixelCrop.width,
    pixelCrop.height
  );

  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Canvas is empty"));
    }, "image/png");
  });
}

// --- Helper: Generate 4x2 Grid Image ---
async function generateGridImage(imageBlob: Blob): Promise<Blob> {
  const imageUrl = URL.createObjectURL(imageBlob);
  const image = await createImage(imageUrl);

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("No 2d context");

  const iw = image.width;
  const ih = image.height;

  // 4x2 Grid
  canvas.width = iw * 4;
  canvas.height = ih * 2;

  for (let y = 0; y < 2; y++) {
    for (let x = 0; x < 4; x++) {
      ctx.drawImage(image, x * iw, y * ih, iw, ih);
    }
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        URL.revokeObjectURL(imageUrl);
        resolve(blob);
      } else reject(new Error("Canvas is empty"));
    }, "image/png");
  });
}

// --- Main Page ---

export default function Home() {
  // Application State
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [cropPixels, setCropPixels] = useState<Area | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [resultBlob, setResultBlob] = useState<Blob | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const [isDownloadModalOpen, setIsDownloadModalOpen] = useState(false);
  const [recommendedCrop, setRecommendedCrop] = useState<{ x: number, y: number } | null>(null);
  const [recommendedZoom, setRecommendedZoom] = useState<number>(1);
  const [progress, setProgress] = useState(0);

  // Deployment Wake-up State
  const [isServerReady, setIsServerReady] = useState(false);
  const [wakingTime, setWakingTime] = useState(0);

  // Settings State
  const [settings, setSettings] = useState<{
    sizeId: SizeId;
    bgColor: BgColor;
    outfitType: OutfitType;
  }>({
    sizeId: "2inch_head",
    bgColor: "#FFFFFF",
    outfitType: "suit_male",
  });

  // --- Health Check Polling ---
  React.useEffect(() => {
    let timer: NodeJS.Timeout;
    let counter: NodeJS.Timeout;

    const checkServerStatus = async () => {
      try {
        // 優先讀取環境變數，如果沒有就用本地端 (方便你之後在自己電腦測試)
        const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
          setIsServerReady(true);
          clearInterval(timer);
          clearInterval(counter);
        }
      } catch (err) {
        console.log("Server still waking up...");
      }
    };

    // Initial check
    checkServerStatus();

    // Poll every 2 seconds
    timer = setInterval(checkServerStatus, 2000);

    // Increment counter every second
    counter = setInterval(() => {
      setWakingTime(prev => prev + 1);
    }, 1000);

    return () => {
      clearInterval(timer);
      clearInterval(counter);
    };
  }, []);

  // Aspect Ratio Mapping
  const getAspectRatio = (id: SizeId) => {
    switch (id) {
      case "1inch": return 28 / 35;
      case "2inch_head": return 35 / 45;
      case "2inch_half": return 42 / 47;
      default: return 35 / 45;
    }
  };

  // --- Handlers ---

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const imageDataUrl = await readFile(file);
      console.log('File Change - Setting imageSrc (length):', imageDataUrl.length);
      setImageSrc(imageDataUrl);
      setResultUrl(null);
      setResultBlob(null);

      // Auto-analyze for Smart Crop
      analyzeImage(file);
    }
  };

  const analyzeImage = async (file: File) => {
    console.log("Analyzing face for recommendation...");
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('size_id', settings.sizeId);

      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${API_URL}/analyze-face`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json(); // {x: %, y: %, width: %, height: %}

        // Convert recommendation (%) to react-easy-crop 'crop' state (translation from center)
        const centerX = (data.x + data.width / 2) - 50;
        const centerY = (data.y + data.height / 2) - 50;

        // Zoom heuristic: fits the crop box into the view
        const zoom = Math.max(1, 100 / Math.max(data.width, data.height) * 0.85);

        setRecommendedCrop({ x: centerX, y: centerY });
        setRecommendedZoom(zoom);
        console.log("Smart Crop Applied:", { centerX, centerY, zoom });
      }
    } catch (err) {
      console.error("Face analysis failed:", err);
    }
  };

  const readFile = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.addEventListener("load", () => resolve(reader.result as string));
      reader.readAsDataURL(file);
    });
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsHovering(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      const imageDataUrl = await readFile(file);
      console.log('Drop - Setting imageSrc (length):', imageDataUrl.length);
      setImageSrc(imageDataUrl);
      setResultUrl(null);
      setResultBlob(null);

      // Auto-analyze for Smart Crop
      analyzeImage(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsHovering(true);
  };

  const handleDragLeave = () => setIsHovering(false);

  const handleGenerate = async () => {
    if (!imageSrc || !cropPixels) return;

    setIsGenerating(true);
    setProgress(0);

    // Simulated Progress logic: 0 to 90 over 30 seconds
    const startTime = Date.now();
    const duration = 30000; // 30 seconds to reach 90%

    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      if (elapsed < duration) {
        // Ease-out curve: progress = 90 * (1 - (1 - t)^2)
        const t = elapsed / duration;
        const easedProgress = Math.floor(90 * (1 - Math.pow(1 - t, 2)));
        setProgress(easedProgress);
      } else {
        // Hold at 99% until done
        setProgress(99);
      }
    }, 200);

    try {
      const croppedBlob = await getCroppedImg(imageSrc, cropPixels);
      const formData = new FormData();
      formData.append("file", croppedBlob, "input.png");
      formData.append("size_id", settings.sizeId);
      formData.append("bg_color", settings.bgColor);
      formData.append("outfit_type", settings.outfitType);

      console.log("Calling backend...");
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
      console.log("Calling backend at:", API_URL); // 方便除錯
      const response = await fetch(`${API_URL}/generate`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Generation failed");
      }

      const blob = await response.blob();
      setProgress(100);
      setResultBlob(blob);
      const url = URL.createObjectURL(blob);
      setResultUrl(url);

    } catch (error: any) {
      console.error(error);
      alert(`Failed to generate image: ${error.message}`);
    } finally {
      clearInterval(progressInterval);
      setTimeout(() => {
        setIsGenerating(false);
      }, 500); // Give time to see 100%
    }
  };

  const downloadSingle = () => {
    if (!resultBlob) return;
    saveAs(resultBlob, `id_photo_1x1_${settings.sizeId}.png`);
  };

  const downloadGrid = async () => {
    if (!resultBlob) return;
    const gridBlob = await generateGridImage(resultBlob);
    saveAs(gridBlob, `id_photo_grid_4x2.png`);
  };

  const downloadZip = async () => {
    if (!resultBlob) return;
    const zip = new JSZip();
    zip.file("ID_Photo_Original.png", resultBlob);

    const gridBlob = await generateGridImage(resultBlob);
    zip.file("ID_Photo_Grid_4x2.png", gridBlob);

    const content = await zip.generateAsync({ type: "blob" });
    saveAs(content, "ID_Photos_All.zip");
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen w-full bg-cream-50 font-sans text-stone-dark relative overflow-x-hidden">

      {/* WAKING UP OVERLAY */}
      {!isServerReady && (
        <div className="fixed inset-0 z-[200] flex items-center justify-center bg-[#faf9f6]/80 backdrop-blur-md animate-fade-in px-6">
          <div className="bg-white p-12 rounded-[3rem] shadow-2xl border border-cream-200 flex flex-col items-center gap-8 max-w-lg text-center">
            <div className="p-8 bg-cream-accent rounded-full animate-bounce-slow text-cream-text shadow-inner">
              <svg viewBox="0 0 24 24" className="w-16 h-16 fill-current">
                <path d="M2.5 19h15V9.5h-15V19zM19 12h2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2h-2v-6zM1 18h18v1H1v-1zM5 15.5h10v1H5v-1zM4 6.5s1-1 2 0 1 1 2 0 1-1 2 0 1 1 2 0M6 4.5s1-1 2 0 1 1 2 0 1-1 2 0 1 1 2 0" />
              </svg>
            </div>
            <div className="space-y-3">
              <h2 className="text-3xl font-black text-stone-dark tracking-tight">系統暖機中...</h2>
              <p className="text-lg text-stone-dark/70 font-medium">
                由於使用免費雲端資源，首次啟動需等待約 60 秒，請稍候 ☕
              </p>
            </div>
            <div className="w-full bg-cream-100 rounded-full h-3 overflow-hidden shadow-inner">
              <div
                className="bg-stone-dark h-full transition-all duration-500 ease-linear rounded-full"
                style={{ width: `${Math.min((wakingTime / 60) * 100, 100)}%` }}
              ></div>
            </div>
            <p className="text-sm font-bold text-stone-dark/40 tracking-widest uppercase">
              (已等待: <span className="text-stone-dark">{wakingTime}</span> 秒)
            </p>
          </div>
        </div>
      )}

      {/* Main Interaction Area with Anti-Fumble */}
      <div className={`flex flex-col lg:flex-row min-h-full w-full transition-all duration-500 ${isGenerating ? "opacity-30 pointer-events-none grayscale cursor-not-allowed select-none" : ""}`}>

        {/* LEFT PANEL (70% on desktop, 100% on mobile) */}
        <div className="w-full lg:w-[70%] lg:min-h-screen relative p-6 md:p-12 flex flex-col items-center justify-center">

          {/* Result Overlay */}
          {resultUrl ? (
            <div className="relative flex flex-col items-center animate-fade-in gap-8 w-full max-w-2xl">
              <h2 className="text-3xl font-bold tracking-tight text-stone-dark mb-2">✨ 生成結果</h2>
              <div className="relative shadow-soft-xl rounded-3xl overflow-hidden border-4 border-white w-full flex justify-center bg-white">
                <img src={resultUrl} alt="Generated ID" className="max-h-[70vh] md:max-h-[600px] object-contain" />
              </div>
              <div className="flex flex-col sm:flex-row gap-4 w-full justify-center">
                <button
                  onClick={() => setIsDownloadModalOpen(true)}
                  className="flex items-center gap-2 bg-stone-dark text-cream-50 px-8 py-4 rounded-3xl font-bold shadow-lg hover:bg-stone-500 transition-all active:scale-95"
                >
                  <Download className="w-5 h-5" /> 下載證件照
                </button>
                <button
                  onClick={() => {
                    setResultUrl(null);
                    setResultBlob(null);
                  }}
                  className="flex items-center gap-2 bg-white text-stone-dark px-8 py-4 rounded-3xl font-bold shadow-sm hover:bg-gray-50 border border-cream-200 transition-all"
                >
                  <RefreshCcw className="w-5 h-5" /> 重新調整
                </button>
              </div>
            </div>
          ) : (
            /* WORKSPACE (Upload or Crop) */
            <div className="w-full max-w-5xl flex-1 relative flex flex-col">

              {!imageSrc ? (
                // Empty State: Upload
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  className={`flex-1 flex flex-col items-center justify-center border-4 border-dashed rounded-[3rem] transition-all duration-300 ${isHovering
                    ? "border-cream-400 bg-cream-100 shadow-soft-xl scale-[1.02]"
                    : "border-cream-200 bg-white hover:bg-cream-50 hover:border-cream-300"
                    }`}
                >
                  <div className="p-10 bg-cream-100 rounded-full mb-8 shadow-inner text-cream-400">
                    <Upload className="w-16 h-16" />
                  </div>
                  <h2 className="text-3xl font-bold text-stone-dark mb-3 tracking-wide">拖放照片或點擊上傳</h2>
                  <p className="text-stone-dark/60 mb-10 text-lg">支援 JPG, PNG 格式 (建議使用清晰正面照)</p>
                  <label className="bg-gradient-to-r from-orange-100 to-amber-100 text-stone-dark border border-cream-300 font-bold py-4 px-12 rounded-3xl cursor-pointer shadow-sm hover:shadow-md transition-all active:scale-95 hover:-translate-y-1">
                    選擇照片 (Select Photo)
                    <input type="file" accept="image/*" className="hidden" onChange={onFileChange} />
                  </label>

                  {/* Upload Guidelines Section */}
                  <div className="mt-12 w-full max-w-xl bg-amber-50/50 border border-amber-100 rounded-[2rem] p-6 text-left shadow-sm">
                    <div className="flex items-center gap-3 mb-4 text-amber-600">
                      <AlertCircle className="w-5 h-5" />
                      <h3 className="font-bold tracking-wide">拍攝規範提示</h3>
                    </div>
                    <ul className="grid grid-cols-1 md:grid-cols-2 gap-3 text-stone-dark/70 text-sm font-medium">
                      <li className="flex items-center gap-2 bg-white/60 p-2 rounded-xl">👀 <span className="text-stone-dark font-bold">雙眼直視</span>：請直視鏡頭</li>
                      <li className="flex items-center gap-2 bg-white/60 p-2 rounded-xl">👂 <span className="text-stone-dark font-bold">露出五官</span>：眉毛與耳朵完整露出</li>
                      <li className="flex items-center gap-2 bg-amber-100/40 p-2 rounded-xl border border-amber-200/50 col-span-full">
                        <span className="flex items-center gap-2">🚫 <span className="text-amber-700 font-black underline underline-offset-4">移除飾品 (最重要)</span>：請取下眼鏡、耳環、項鍊</span>
                      </li>
                      <li className="flex items-center gap-2 bg-white/60 p-2 rounded-xl">💡 <span className="text-stone-dark font-bold">光線充足</span>：避免臉部陰影</li>
                    </ul>
                  </div>
                </div>
              ) : (
                // Loaded State: Cropper
                <div className="flex-1 relative flex flex-col gap-6">
                  <div className="flex items-center justify-between px-6">
                    <h2 className="text-2xl font-bold text-stone-dark">調整裁切範圍 (Adjust Crop)</h2>
                    <button onClick={() => setImageSrc(null)} className="text-stone-dark/60 hover:text-red-400 flex items-center gap-2 font-medium transition-colors bg-white px-4 py-2 rounded-full shadow-sm">
                      <XCircle className="w-5 h-5" /> 清除 (Clear)
                    </button>
                  </div>

                  <div className="flex-1 relative rounded-[32px] overflow-hidden shadow-soft-xl border-4 border-white bg-black h-[50vh] min-h-[350px] md:h-[60vh] md:min-h-[400px] w-full">
                    {(() => { console.log('Crop Image Source:', imageSrc); return null; })()}
                    <CropperComponent
                      imageSrc={imageSrc}
                      aspectRatio={getAspectRatio(settings.sizeId)}
                      onCropComplete={(pixels: any) => setCropPixels(pixels)}
                      initialCrop={recommendedCrop || undefined}
                      initialZoom={recommendedZoom}
                    />
                  </div>
                  <p className="text-center text-stone-dark/60 text-sm font-medium tracking-wide">
                    您可以縮放與移動照片，確保臉部對齊網格中心
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* RIGHT PANEL (30% on desktop, 100% on mobile) */}
        <div className="w-full lg:w-[30%] lg:h-screen lg:sticky lg:top-0 z-20">
          <Sidebar
            settings={settings}
            setSettings={setSettings}
            onGenerate={handleGenerate}
            isGenerating={isGenerating}
            hasImage={!!imageSrc}
          />
        </div>
      </div>

      {/* GENERATING OVERLAY */}
      {isGenerating && (
        <div className="absolute inset-0 z-[100] flex items-center justify-center">
          <div className="bg-white/80 backdrop-blur-xl px-12 py-10 rounded-[3rem] shadow-2xl flex flex-col items-center gap-8 animate-fade-in border-2 border-cream-400">
            {/* Circular Progress Bar */}
            <div className="relative w-28 h-28">
              <svg className="w-full h-full" viewBox="0 0 100 100">
                {/* Background Track */}
                <circle
                  cx="50" cy="50" r="45"
                  className="stroke-stone-dark/5"
                  strokeWidth="8"
                  fill="none"
                />
                {/* Progress Path */}
                <circle
                  cx="50" cy="50" r="45"
                  className="stroke-[#4CAF50] transition-all duration-300 ease-out"
                  strokeWidth="8"
                  fill="none"
                  strokeLinecap="round"
                  style={{
                    strokeDasharray: 282.7,
                    strokeDashoffset: 282.7 - (282.7 * progress) / 100,
                  }}
                />
              </svg>
              {/* Percentage Text */}
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-black text-stone-dark">{progress}%</span>
              </div>
            </div>

            <div className="text-center">
              <p className="text-2xl font-black text-stone-dark tracking-tighter">AI 拼命生成中...</p>
              <p className="text-stone-dark/40 font-bold text-sm">請稍待片刻，這通常需要 3~5 分鐘</p>
            </div>
          </div>
        </div>
      )}

      {/* DOWNLOAD MODAL */}
      {isDownloadModalOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-stone-dark/40 backdrop-blur-md animate-fade-in">
          <div className="bg-[#faf9f6] w-[500px] rounded-[2.5rem] shadow-2xl relative p-10 flex flex-col items-center gap-8 border border-white/50">
            <button
              onClick={() => setIsDownloadModalOpen(false)}
              className="absolute top-6 right-6 text-stone-dark/40 hover:text-stone-dark transition-colors p-2"
            >
              <X className="w-6 h-6" />
            </button>

            <div className="text-center">
              <h3 className="text-2xl font-bold text-stone-dark mb-2">📥 下載您的證件照</h3>
              <p className="text-stone-dark/60 text-sm">請選擇您需要的下載格式</p>
            </div>

            <div className="w-full flex flex-col gap-4">
              <button
                onClick={downloadSingle}
                className="flex items-center gap-4 w-full p-4 rounded-3xl bg-white border border-cream-200 text-stone-dark hover:bg-cream-100 hover:border-cream-300 transition-all font-bold shadow-sm"
              >
                <div className="p-3 bg-cream-100 rounded-2xl text-stone-dark">
                  <User className="w-6 h-6" />
                </div>
                <div className="text-left">
                  <p className="text-lg">單張原圖 (1x1)</p>
                  <p className="text-xs text-stone-dark/50 font-medium">原始解析度裁切照片</p>
                </div>
              </button>

              <button
                onClick={downloadGrid}
                className="flex items-center gap-4 w-full p-4 rounded-3xl bg-white border border-cream-200 text-stone-dark hover:bg-cream-100 hover:border-cream-300 transition-all font-bold shadow-sm"
              >
                <div className="p-3 bg-cream-100 rounded-2xl text-stone-dark">
                  <Grid className="w-6 h-6" />
                </div>
                <div className="text-left">
                  <p className="text-lg">排版列印用 (4x2)</p>
                  <p className="text-xs text-stone-dark/50 font-medium">適合 4x6 吋相紙拼貼</p>
                </div>
              </button>

              <button
                onClick={downloadZip}
                className="flex items-center gap-4 w-full p-5 rounded-3xl bg-gradient-to-r from-orange-200 to-amber-200 text-stone-dark transition-all hover:shadow-lg hover:scale-[1.02] active:scale-[0.98] font-black shadow-md border border-orange-300/30"
              >
                <div className="p-3 bg-white/40 rounded-2xl">
                  <FileArchive className="w-7 h-7" />
                </div>
                <div className="text-left">
                  <p className="text-xl">小孩才做選擇 (ZIP全都要)</p>
                  <p className="text-xs opacity-70">打包所有版本一次下載</p>
                </div>
              </button>
            </div>

            <p className="text-[10px] text-stone-dark/30 tracking-widest uppercase font-bold">Pro-ID Gen Premium Export</p>
          </div>
        </div>
      )}
    </div>
  );
}
