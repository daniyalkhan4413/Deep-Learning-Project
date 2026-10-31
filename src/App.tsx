import { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';
import { 
  ShieldCheck, AlertTriangle, ShieldAlert, Upload, RefreshCw, Eye, 
  CheckCircle2, XCircle, Terminal, ChevronDown, ChevronUp,
  Database, FileText, Video, Share2, ExternalLink, BarChart3, Layers, Cpu,
  HelpCircle, Camera, X
} from 'lucide-react';

function GithubIcon({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24">
      <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  );
}

interface PredictionResult {
  tier: 'approved' | 'review' | 'rejected';
  confidence: number;
  label: string;
  recommendation: string;
  rawLogits: number[];
  probReal: number;
  probSpoof: number;
  latencyMs: number;
}

const PROJECT_LINKS = [
  { label: 'GitHub Repository', sub: 'Code, notebooks, ONNX model weights', icon: GithubIcon, url: 'https://github.com' },
  { label: 'Dataset (Datacluster Labs)', sub: 'Kaggle Real-World vs. Moiré dataset', icon: Database, url: 'https://www.kaggle.com/datasets/dataclusterlabs/real-world-and-moire-pattern-classification' },
  { label: 'Project Report', sub: 'Methodology & forensic anti-spoofing report', icon: FileText, url: '#' },
  { label: 'Presentation (PPT)', sub: 'Architecture & evaluation slide deck', icon: FileText, url: '#' },
  { label: 'YouTube Video', sub: 'Technical voiceover walkthrough & demo', icon: Video, url: '#' },
  { label: 'LinkedIn Post', sub: 'Project launch & announcement', icon: Share2, url: '#' },
];

const BENCHMARK_RESULTS = [
  { 
    model: 'Custom Baseline CNN (4 Conv Blocks)', 
    labAcc: '84.2%', 
    realWorldAcc: '50.0%', 
    note: 'Collapses to chance level (50%) on real-world test distributions' 
  },
  { 
    model: 'MobileNetV3-Small (Naive 50% Cutoff)', 
    labAcc: '89.4%', 
    realWorldAcc: '61.2%', 
    note: 'Vulnerable to background textures (mesh fabric, carbon weave, desk grain)' 
  },
  { 
    model: 'MobileNetV3-Small (ROI Inner-Crop + 3-Tier Threshold)', 
    labAcc: '88.7%', 
    realWorldAcc: '88.2%', 
    note: 'Deployed model: isolates document body and prevents perimeter false alarms' 
  },
];

export default function App() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<string>('Initializing...');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [showLogs, setShowLogs] = useState<boolean>(true);
  
  // Camera States
  const [isCameraOpen, setIsCameraOpen] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isInitializingRef = useRef<boolean>(false);

  // Initialize ONNX WebAssembly session
  useEffect(() => {
    if (isInitializingRef.current || session) return;
    isInitializingRef.current = true;

    async function loadModel() {
      try {
        setErrorMessage(null);
        setLoadingStatus('Configuring WASM runtime...');

        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        ort.env.wasm.numThreads = 1;

        console.log('%c[KYC Shield] Loading ONNX model binary...', 'color: #3b82f6; font-weight: bold;');
        setLoadingStatus('Downloading model weights (~9 MB)...');
        const response = await fetch('/kyc_fraud_detector.onnx');
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: Ensure kyc_fraud_detector.onnx is located directly inside public/`);
        }

        const modelBuffer = await response.arrayBuffer();

        setLoadingStatus('Compiling neural network session...');
        const sess = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ['wasm']
        });

        console.log('%c[KYC Shield] ONNX Model compiled successfully!', 'color: #10b981; font-weight: bold;');
        setSession(sess);
        setLoadingStatus('Ready');
      } catch (err: unknown) {
        console.error('[KYC Shield] Model initialization error:', err);
        const msg = err instanceof Error ? err.message : String(err);
        setErrorMessage(msg);
        setLoadingStatus('Failed');
      }
    }

    loadModel();
  }, [session]);

  // Clean up camera on unmount
  useEffect(() => {
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    setIsCameraOpen(true);
    setErrorMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Camera error:", err);
      setErrorMessage("Camera access denied or unavailable. Please upload a file instead.");
      setIsCameraOpen(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraOpen(false);
  };

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg');
        setImagePreview(dataUrl);
        stopCamera();
        setResult(null);
        runInference(dataUrl);
      }
    }
  };

  const runInference = async (imgUrl: string) => {
    if (!session) {
      setErrorMessage('Model is not initialized yet.');
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);
    const startTime = performance.now();

    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = imgUrl;
      await img.decode();

      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Could not access canvas 2D context');

      // ROI Core Crop (Center 70%)
      const naturalWidth = img.naturalWidth || img.width;
      const naturalHeight = img.naturalHeight || img.height;
      const minDimension = Math.min(naturalWidth, naturalHeight);

      const cropSize = Math.floor(minDimension * 0.70);
      const cropX = Math.floor((naturalWidth - cropSize) / 2);
      const cropY = Math.floor((naturalHeight - cropSize) / 2);

      ctx.drawImage(
        img,
        cropX, cropY, cropSize, cropSize,
        0, 0, 224, 224
      );

      const imgData = ctx.getImageData(0, 0, 224, 224).data;
      const float32Data = new Float32Array(3 * 224 * 224);
      const mean = [0.485, 0.456, 0.406];
      const std = [0.229, 0.224, 0.225];

      for (let i = 0; i < 224 * 224; i++) {
        const r = imgData[i * 4] / 255.0;
        const g = imgData[i * 4 + 1] / 255.0;
        const b = imgData[i * 4 + 2] / 255.0;

        float32Data[i] = (r - mean[0]) / std[0];
        float32Data[224 * 224 + i] = (g - mean[1]) / std[1];
        float32Data[2 * 224 * 224 + i] = (b - mean[2]) / std[2];
      }

      const inputName = session.inputNames[0];
      const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);

      // 600ms delay to display the scanning beam animation
      await new Promise(resolve => setTimeout(resolve, 600));

      const outputs = await session.run({ [inputName]: inputTensor });

      const outputName = session.outputNames[0];
      const logits = outputs[outputName].data as Float32Array;
      const latency = Math.round(performance.now() - startTime - 600);

      const maxLogit = Math.max(logits[0], logits[1]);
      const exp0 = Math.exp(logits[0] - maxLogit);
      const exp1 = Math.exp(logits[1] - maxLogit);
      const prob0 = exp0 / (exp0 + exp1); 
      const prob1 = exp1 / (exp0 + exp1); 

      let tier: 'approved' | 'review' | 'rejected';
      let label: string;
      let recommendation: string;
      let confidence: number;

      if (prob1 >= 0.65) {
        tier = 'rejected';
        label = 'Screen Replay Attack Detected';
        recommendation = 'Reject upload. Require live video liveness verification.';
        confidence = Math.round(prob1 * 100);
      } else if (prob1 >= 0.35) {
        tier = 'review';
        label = 'Borderline Texture / Ambiguous Scan';
        recommendation = 'Retake required: Place document flat on a plain, non-textured surface.';
        confidence = Math.round(Math.max(prob0, prob1) * 100);
      } else {
        tier = 'approved';
        label = 'Genuine Physical Document';
        recommendation = 'Approve document for identity verification and OCR extraction.';
        confidence = Math.round(prob0 * 100);
      }

      console.group('%c🔍 [KYC Shield] Document Diagnostics', 'color: #38bdf8; font-weight: bold;');
      console.log('⏱️ Latency:', `${latency} ms`);
      console.log('📊 Logits [Real, Spoof]:', [logits[0].toFixed(4), logits[1].toFixed(4)]);
      console.log('📈 Softmax:', { Real: `${(prob0 * 100).toFixed(2)}%`, Spoof: `${(prob1 * 100).toFixed(2)}%` });
      console.log(`🏁 Policy Tier: ${tier.toUpperCase()} (${confidence}%)`);
      console.groupEnd();

      setResult({
        tier,
        confidence,
        label,
        recommendation,
        rawLogits: [Number(logits[0].toFixed(4)), Number(logits[1].toFixed(4))],
        probReal: Number((prob0 * 100).toFixed(2)),
        probSpoof: Number((prob1 * 100).toFixed(2)),
        latencyMs: latency
      });
    } catch (err: unknown) {
      console.error('[KYC Shield] Inference error:', err);
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(`Inference failed: ${msg}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        const url = reader.result as string;
        setImagePreview(url);
        setResult(null);
        runInference(url);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 flex flex-col items-center py-12 px-4 sm:px-6">
      
      {/* CSS Injection for Scanning Animation */}
      <style>{`
        @keyframes scanline {
          0% { top: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        .animate-scanline {
          animation: scanline 1.5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        }
        .fade-in-up {
          animation: fadeInUp 0.5s ease-out forwards;
        }
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(15px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div className="max-w-3xl w-full space-y-10">

        {/* Header */}
        <header className="text-center space-y-3 fade-in-up">
          <div className="inline-flex items-center gap-1.5 bg-blue-500/10 text-blue-400 px-3.5 py-1 rounded-full text-xs font-semibold border border-blue-500/20 shadow-[0_0_15px_rgba(59,130,246,0.15)]">
            <ShieldCheck className="w-4 h-4" /> KYC Identity Security Engine
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
            KYC Shield: Screen Replay Detector
          </h1>
          <p className="text-gray-400 text-sm max-w-xl mx-auto">
            Client-side neural network analyzing Moiré patterns, LCD subpixel grids, and monitor recapture fraud with ROI background suppression.
          </p>
        </header>

        {/* Error Banner */}
        {errorMessage && (
          <div className="p-4 bg-red-950/60 border border-red-700/80 rounded-xl text-red-200 text-xs flex items-start gap-2.5 shadow-lg animate-pulse">
            <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1 font-mono">{errorMessage}</div>
          </div>
        )}

        {/* Upload / Camera Terminal */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-2xl fade-in-up" style={{ animationDelay: '0.1s' }}>
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-700 rounded-xl p-6 transition-all duration-300 hover:border-blue-500/50 hover:bg-blue-900/5 relative overflow-hidden">
            
            {/* Live Camera View */}
            {isCameraOpen ? (
              <div className="w-full flex flex-col items-center">
                <div className="relative w-full max-w-sm rounded-lg overflow-hidden border border-gray-700 shadow-lg">
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    playsInline 
                    muted 
                    className="w-full h-64 object-cover"
                  />
                  <div className="absolute inset-0 border-[4px] border-white/20 m-6 rounded-xl pointer-events-none">
                    <div className="absolute inset-0 border-[2px] border-dashed border-white/40 rounded-xl"></div>
                  </div>
                </div>
                <div className="flex gap-4 mt-6">
                  <button 
                    onClick={stopCamera}
                    className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg text-sm font-medium transition-transform hover:scale-105 flex items-center gap-2 cursor-pointer"
                  >
                    <X className="w-4 h-4" /> Cancel
                  </button>
                  <button 
                    onClick={capturePhoto}
                    className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-transform hover:scale-105 shadow-[0_0_15px_rgba(37,99,235,0.4)] flex items-center gap-2 cursor-pointer"
                  >
                    <Camera className="w-4 h-4" /> Capture Photo
                  </button>
                </div>
              </div>
            ) : (
              /* Image Preview or Upload Prompt */
              <>
                {imagePreview ? (
                  <div className="relative rounded-lg overflow-hidden mb-5 shadow-lg group">
                    <img src={imagePreview} alt="Target Document" className="h-60 object-contain" />
                    
                    {/* CSS Scanning Animation Overlay */}
                    {isProcessing && (
                      <>
                        <div className="absolute inset-0 bg-blue-500/10 z-10"></div>
                        <div className="absolute left-0 right-0 h-1 bg-blue-400 shadow-[0_0_12px_#60a5fa] animate-scanline z-20"></div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center py-8 text-gray-400 group">
                    <div className="bg-gray-800 p-4 rounded-full mb-4 group-hover:scale-110 transition-transform duration-300">
                      <Upload className="w-8 h-8 text-gray-400 group-hover:text-blue-400 transition-colors" />
                    </div>
                    <p className="font-medium text-gray-300">Upload or capture document</p>
                    <p className="text-xs text-gray-500 mt-1.5 text-center max-w-xs">
                      Supports direct physical ID captures and screen presentation tests.
                    </p>
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileUpload}
                />

                <div className="flex flex-wrap gap-3 justify-center">
                  {!session ? (
                     <button disabled className="flex items-center gap-2 px-6 py-2.5 bg-blue-600/50 text-white rounded-lg font-medium text-sm cursor-not-allowed">
                       <RefreshCw className="w-4 h-4 animate-spin" /> {loadingStatus}
                     </button>
                  ) : (
                    <>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isProcessing}
                        className="flex items-center gap-2 px-5 py-2.5 bg-gray-800 hover:bg-gray-700 text-white rounded-lg font-medium transition-all duration-300 hover:scale-105 disabled:opacity-50 text-sm shadow-md cursor-pointer"
                      >
                        <Upload className="w-4 h-4" /> {imagePreview ? 'Upload New' : 'Upload File'}
                      </button>

                      {!imagePreview && (
                        <button
                          onClick={startCamera}
                          disabled={isProcessing}
                          className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-all duration-300 hover:scale-105 disabled:opacity-50 text-sm shadow-[0_0_15px_rgba(37,99,235,0.3)] cursor-pointer"
                        >
                          <Camera className="w-4 h-4" /> Open Camera
                        </button>
                      )}

                      {imagePreview && (
                        <button
                          onClick={() => runInference(imagePreview)}
                          disabled={isProcessing}
                          className="flex items-center gap-2 px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium transition-all duration-300 hover:scale-105 disabled:opacity-50 text-sm shadow-[0_0_15px_rgba(5,150,105,0.4)] cursor-pointer"
                        >
                          {isProcessing ? (
                            <>
                              <RefreshCw className="w-4 h-4 animate-spin" /> Analyzing Subpixel Grids...
                            </>
                          ) : (
                            <>
                              <Eye className="w-4 h-4" /> Run Forensic Scan
                            </>
                          )}
                        </button>
                      )}
                    </>
                  )}
                </div>
              </>
            )}
          </div>
        </div>

        {/* 3-Tier Enterprise Report Card */}
        {result && (
          <div className={`border rounded-2xl p-6 shadow-2xl fade-in-up ${
            result.tier === 'rejected'
              ? 'bg-red-950/20 border-red-800/60 shadow-[0_4px_30px_rgba(220,38,38,0.1)]'
              : result.tier === 'review'
              ? 'bg-amber-950/20 border-amber-700/60 shadow-[0_4px_30px_rgba(217,119,6,0.1)]'
              : 'bg-emerald-950/20 border-emerald-800/60 shadow-[0_4px_30px_rgba(5,150,105,0.1)]'
          }`}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                {result.tier === 'rejected' && <ShieldAlert className="w-7 h-7 text-red-400 flex-shrink-0" />}
                {result.tier === 'review' && <HelpCircle className="w-7 h-7 text-amber-400 flex-shrink-0" />}
                {result.tier === 'approved' && <ShieldCheck className="w-7 h-7 text-emerald-400 flex-shrink-0" />}
                
                <div>
                  <h3 className="font-bold text-white text-base">
                    {result.tier === 'rejected' && 'Fraud Alert: Replay Attack Confirmed'}
                    {result.tier === 'review' && 'Texture Uncertainty / Surface Warning'}
                    {result.tier === 'approved' && 'Authentic Physical Document Passed'}
                  </h3>
                  <p className="text-xs text-gray-400 mt-0.5">{result.label}</p>
                </div>
              </div>
              <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                result.tier === 'rejected'
                  ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                  : result.tier === 'review'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              }`}>
                {result.confidence}% Match
              </span>
            </div>

            <div className="bg-gray-900/90 border border-gray-800 rounded-lg p-4 space-y-3 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">High-Frequency Moiré Grid:</span>
                <span className={`flex items-center gap-1.5 font-semibold ${
                  result.tier === 'rejected' ? 'text-red-400' : result.tier === 'review' ? 'text-amber-400' : 'text-emerald-400'
                }`}>
                  {result.tier === 'rejected' && <><XCircle className="w-3.5 h-3.5" /> High-Density Interference</>}
                  {result.tier === 'review' && <><AlertTriangle className="w-3.5 h-3.5" /> Borderline Periodic Texture</>}
                  {result.tier === 'approved' && <><CheckCircle2 className="w-3.5 h-3.5" /> Clean Analog Paper Grain</>}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-400">Decision Tier:</span>
                <span className={`font-mono font-semibold uppercase ${
                  result.tier === 'rejected' ? 'text-red-400' : result.tier === 'review' ? 'text-amber-400' : 'text-emerald-400'
                }`}>
                  {result.tier} (Spoof P = {result.probSpoof}%)
                </span>
              </div>

              <div className="pt-2.5 border-t border-gray-800">
                <span className="text-gray-400 block mb-1.5">KYC Policy Recommendation:</span>
                <p className={`font-medium text-[13px] leading-relaxed ${
                  result.tier === 'rejected' ? 'text-red-300' : result.tier === 'review' ? 'text-amber-300' : 'text-emerald-300'
                }`}>
                  {result.recommendation}
                </p>
              </div>
            </div>

            {/* In-UI Telemetry Drawer */}
            <div className="mt-4 border-t border-gray-800/80 pt-3">
              <button
                onClick={() => setShowLogs(!showLogs)}
                className="flex items-center justify-between w-full text-xs text-gray-400 hover:text-gray-200 transition font-mono cursor-pointer"
              >
                <span className="flex items-center gap-1.5 font-semibold text-blue-400">
                  <Terminal className="w-3.5 h-3.5" /> Live Model Telemetry & Logits
                </span>
                {showLogs ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>

              {showLogs && (
                <div className="mt-3 p-4 bg-black/70 border border-gray-800 rounded-lg font-mono text-[11px] space-y-2 text-gray-300 shadow-inner">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Inference Latency:</span>
                    <span className="text-amber-400 font-semibold">{result.latencyMs} ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Raw Logits [Real, Spoof]:</span>
                    <span className="text-blue-300">[{result.rawLogits[0]}, {result.rawLogits[1]}]</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Calculated P(Real):</span>
                    <span className="text-emerald-400">{result.probReal}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Calculated P(Spoof):</span>
                    <span className="text-red-400">{result.probSpoof}%</span>
                  </div>
                  <div className="flex justify-between pt-2 mt-1 border-t border-gray-800 text-[10px] text-gray-400">
                    <span>Engine: WASM SIMD | Target: ROI 70% Core Isolation</span>
                    <span>Tensor: [1, 3, 224, 224]</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Section: Project Links */}
        <section className="space-y-4 fade-in-up" style={{ animationDelay: '0.2s' }}>
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-300">
            <ExternalLink className="w-4 h-4 text-blue-400" /> Project Deliverables & Links
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {PROJECT_LINKS.map((item) => {
              const Icon = item.icon;
              return (
                <a
                  key={item.label}
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                  className="p-4 bg-gray-900 border border-gray-800 hover:border-blue-500/50 rounded-xl transition-all duration-300 hover:-translate-y-1 group flex flex-col justify-between shadow-lg hover:shadow-[0_4px_20px_rgba(59,130,246,0.15)]"
                >
                  <div className="flex items-center justify-between mb-3">
                    <Icon className="w-5 h-5 text-blue-400 group-hover:scale-110 transition-transform duration-300" />
                    <ExternalLink className="w-3.5 h-3.5 text-gray-600 group-hover:text-blue-400 transition-colors" />
                  </div>
                  <div>
                    <div className="font-semibold text-xs text-white group-hover:text-blue-400 transition-colors">{item.label}</div>
                    <div className="text-[11px] text-gray-400 mt-1">{item.sub}</div>
                  </div>
                </a>
              );
            })}
          </div>
        </section>

        {/* Section: Results & Generalization Gap */}
        <section className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-4 shadow-xl fade-in-up" style={{ animationDelay: '0.3s' }}>
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-200">
            <BarChart3 className="w-4 h-4 text-blue-400" /> Experimental Results & The Generalization Gap
          </div>
          
          <div className="overflow-x-auto rounded-lg border border-gray-800">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-gray-800/50 text-gray-300 border-b border-gray-800">
                  <th className="py-3 px-4 font-semibold">Model Architecture</th>
                  <th className="py-3 px-4 font-semibold">Lab Moiré Test Accuracy</th>
                  <th className="py-3 px-4 font-semibold">Real-World In-The-Wild Accuracy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60 font-mono bg-gray-900">
                {BENCHMARK_RESULTS.map((row, idx) => (
                  <tr key={idx} className="hover:bg-gray-800/30 transition-colors duration-200">
                    <td className="py-3.5 px-4 font-sans font-medium text-gray-300">
                      {row.model}
                      <span className="block text-[10px] text-gray-500 font-mono mt-1">{row.note}</span>
                    </td>
                    <td className="py-3.5 px-4 text-blue-400 font-semibold">{row.labAcc}</td>
                    <td className="py-3.5 px-4">
                      <span className={`px-2.5 py-1 rounded-md text-[11px] font-bold ${
                        parseFloat(row.realWorldAcc) <= 65.0 
                          ? 'bg-red-500/20 text-red-400 border border-red-500/30' 
                          : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      }`}>
                        {row.realWorldAcc}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-4 bg-blue-950/20 border border-blue-800/40 rounded-lg text-xs text-blue-300 leading-relaxed shadow-inner">
            <strong>Edge Case Resolution:</strong> Real-world false rejections are predominantly driven by background fabrics (car seats, weaves) and downsampling aliasing. Introducing a <strong>70% Region of Interest (ROI) core crop</strong> removes perimeter background noise, while a <strong>3-tier decision policy</strong> routes ambiguous edge cases ($35\% \le P \lt 65\%$) to a user retry rather than an outright fraud rejection.
          </div>
        </section>

        {/* Section: Dataset & Model Specifications */}
        <section className="grid grid-cols-1 sm:grid-cols-2 gap-4 fade-in-up" style={{ animationDelay: '0.4s' }}>
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-4 shadow-xl hover:shadow-[0_4px_25px_rgba(59,130,246,0.05)] transition-shadow duration-300">
            <div className="flex items-center gap-2 text-sm font-semibold text-gray-200">
              <Layers className="w-4 h-4 text-blue-400" /> Dataset Specifications
            </div>
            <ul className="text-xs text-gray-400 space-y-3 leading-relaxed">
              <li className="pl-3 border-l-2 border-gray-700">
                <strong className="text-gray-300 block mb-0.5">Benchmark Dataset:</strong> Datacluster Labs Real-World vs. Moiré Pattern dataset consisting of paired photographic captures of real documents vs. screen-recaptured presentations.
              </li>
              <li className="pl-3 border-l-2 border-gray-700">
                <strong className="text-gray-300 block mb-0.5">Real-World Evaluation Set:</strong> Self-collected physical documents (bills of lading, government IDs, invoices) captured across varying smartphone cameras, skewed lighting, and complex background surfaces.
              </li>
            </ul>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-4 shadow-xl hover:shadow-[0_4px_25px_rgba(59,130,246,0.05)] transition-shadow duration-300">
            <div className="flex items-center gap-2 text-sm font-semibold text-gray-200">
              <Cpu className="w-4 h-4 text-blue-400" /> Model Pipeline
            </div>
            <ul className="text-xs text-gray-400 space-y-3 leading-relaxed">
              <li className="pl-3 border-l-2 border-gray-700">
                <strong className="text-gray-300 block mb-0.5">Base Architecture:</strong> MobileNetV3-Small pretrained on ImageNet with a binary linear classification head.
              </li>
              <li className="pl-3 border-l-2 border-gray-700">
                <strong className="text-gray-300 block mb-0.5">Preprocessing & Normalization:</strong> 70% ROI inner-core crop to strip border noise, scaled to $224\times 224$ with ImageNet normalization.
              </li>
              <li className="pl-3 border-l-2 border-gray-700">
                <strong className="text-gray-300 block mb-0.5">Inference Runtime:</strong> Exported to ONNX and executed directly in the client browser via WebAssembly (WASM SIMD) with zero server latency.
              </li>
            </ul>
          </div>
        </section>

      </div>
    </main>
  );
}