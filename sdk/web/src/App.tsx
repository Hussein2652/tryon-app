import { ChangeEvent, FormEvent, useMemo, useState } from "react";
import { recommendSize, requestTryOn, deleteTryOn, SizeRecommendPayload } from "./api";

const DEFAULT_SIZE_PAYLOAD: SizeRecommendPayload = {
  brand_id: "ACME",
  category: "upper_top",
  fit: "regular",
  stretch_pct: 0.05,
  inputs: { method: "direct", chest_circ_cm: 96 },
  size_chart: [
    { size: "S", half_chest_cm: 49 },
    { size: "M", half_chest_cm: 52 },
    { size: "L", half_chest_cm: 55 }
  ]
};

export default function App() {
  const [sizePayload, setSizePayload] = useState<SizeRecommendPayload>(DEFAULT_SIZE_PAYLOAD);
  const [sizeResult, setSizeResult] = useState<Record<string, unknown> | null>(null);
  const [sizeLoading, setSizeLoading] = useState(false);
  const [sizeError, setSizeError] = useState<string | null>(null);

  const [userFile, setUserFile] = useState<File | null>(null);
  const [garmentFile, setGarmentFile] = useState<File | null>(null);
  const [tryOnResult, setTryOnResult] = useState<{
    cacheKey: string;
    images: string[];
    confidence: number;
    frameScores: number[];
  } | null>(null);
  const [tryOnLoading, setTryOnLoading] = useState(false);
  const [tryOnIssues, setTryOnIssues] = useState<string | null>(null);

  const poses = useMemo(() => tryOnResult?.images ?? [], [tryOnResult]);

  const handleSizeSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSizeLoading(true);
    setSizeError(null);
    try {
      const data = await recommendSize(sizePayload);
      setSizeResult(data);
    } catch (error: any) {
      setSizeError(error?.response?.data?.detail ?? "Failed to fetch recommendation");
    } finally {
      setSizeLoading(false);
    }
  };

  const handleTryOnSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!userFile || !garmentFile) {
      setTryOnIssues("Upload both user and garment images");
      return;
    }
    setTryOnLoading(true);
    setTryOnIssues(null);
    try {
      const formData = new FormData();
      formData.append("user_photo", userFile);
      formData.append("garment_front", garmentFile);
      formData.append("sku", "SKU123");
      const recommended = (sizeResult as any)?.recommended_size ?? "M";
      formData.append("size", String(recommended));
      const data = await requestTryOn(formData);
      setTryOnResult({
        cacheKey: data.cache_key,
        images: data.images,
        confidence: data.confidence_avg,
        frameScores: data.frame_scores
      });
    } catch (error: any) {
      const detail = error?.response?.data?.detail;
      if (Array.isArray(detail)) {
        setTryOnIssues(detail.map((issue: any) => issue.message ?? JSON.stringify(issue)).join("; "));
      } else {
        setTryOnIssues(detail ?? "Try-on request failed");
      }
    } finally {
      setTryOnLoading(false);
    }
  };

  const handleDeleteCache = async () => {
    if (!tryOnResult) return;
    await deleteTryOn(tryOnResult.cacheKey);
    setTryOnResult(null);
  };

  const bindUserFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setUserFile(file);
  };

  const bindGarmentFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setGarmentFile(file);
  };

  return (
    <div className="app">
      <section className="panel">
        <h2>1. Size Recommendation</h2>
        <form onSubmit={handleSizeSubmit} className="form-row">
          <label>
            Chest (cm)
            <input
              type="number"
              value={sizePayload.inputs.chest_circ_cm ?? ""}
              onChange={(event) => {
                const value = event.target.value;
                setSizePayload((prev) => ({
                  ...prev,
                  inputs: {
                    ...prev.inputs,
                    chest_circ_cm: value ? Number(value) : undefined
                  }
                }));
              }}
              min={60}
              max={140}
            />
          </label>
          <label>
            Stretch %
            <input
              type="number"
              step={0.01}
              value={sizePayload.stretch_pct}
              onChange={(event) =>
                setSizePayload((prev) => ({ ...prev, stretch_pct: Number(event.target.value) }))
              }
            />
          </label>
          <button type="submit" disabled={sizeLoading}>
            {sizeLoading ? "Computing…" : "Recommend Size"}
          </button>
        </form>
        {sizeError && <div className="issues">{sizeError}</div>}
        {sizeResult && (
          <pre>{JSON.stringify(sizeResult, null, 2)}</pre>
        )}
      </section>

      <section className="panel">
        <h2>2. Try-On Preview</h2>
        <form onSubmit={handleTryOnSubmit} className="form-row">
          <label>
            User Photo
            <input type="file" accept="image/*" onChange={bindUserFile} />
          </label>
          <label>
            Garment Front PNG
            <input type="file" accept="image/png" onChange={bindGarmentFile} />
          </label>
          <button type="submit" disabled={tryOnLoading}>
            {tryOnLoading ? "Rendering…" : "Preview"}
          </button>
        </form>
        {tryOnIssues && <div className="issues">{tryOnIssues}</div>}
        {tryOnResult && (
          <div>
            <p>
              Cache key: {tryOnResult.cacheKey} · Confidence: {tryOnResult.confidence.toFixed(2)}
            </p>
            <button type="button" onClick={handleDeleteCache}>
              Delete previews
            </button>
            <div className="poses">
              {poses.map((image, idx) => (
                <div className="pose-card" key={image}>
                  <img src={image} alt={`Pose ${idx + 1}`} />
                  <span>Frame score: {tryOnResult.frameScores[idx].toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
