import { ChangeEvent, FormEvent, useMemo, useState } from "react";
import { recommendSize, requestTryOn, requestTryOnCompare, deleteTryOn, SizeRecommendPayload } from "./api";

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
  const [compareResult, setCompareResult] = useState<{
    sizeA: string;
    sizeB: string;
    setA: { cacheKey: string; images: string[]; frameScores: number[]; confidence: number };
    setB: { cacheKey: string; images: string[]; frameScores: number[]; confidence: number };
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

  const handleCompare = async () => {
    if (!userFile || !garmentFile || !sizeResult) return;
    const recommended = (sizeResult as any)?.recommended_size as string | undefined;
    const nearestAlt = (sizeResult as any)?.nearest_alt?.size as string | undefined;
    if (!recommended || !nearestAlt) return;
    setTryOnLoading(true);
    setTryOnIssues(null);
    try {
      const formData = new FormData();
      formData.append("user_photo", userFile);
      formData.append("garment_front", garmentFile);
      formData.append("sku", "SKU123");
      formData.append("size_a", String(recommended));
      formData.append("size_b", String(nearestAlt));
      const data = await requestTryOnCompare(formData);
      setCompareResult({
        sizeA: data.size_a,
        sizeB: data.size_b,
        setA: {
          cacheKey: data.set_a.cache_key,
          images: data.set_a.images,
          frameScores: data.set_a.frame_scores,
          confidence: data.set_a.confidence_avg
        },
        setB: {
          cacheKey: data.set_b.cache_key,
          images: data.set_b.images,
          frameScores: data.set_b.frame_scores,
          confidence: data.set_b.confidence_avg
        }
      });
    } catch (error: any) {
      const detail = error?.response?.data?.detail;
      setTryOnIssues(detail ?? "Compare sizes request failed");
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
            Category
            <select
              value={sizePayload.category}
              onChange={(e) => setSizePayload((prev) => ({ ...prev, category: e.target.value }))}
            >
              <option value="upper_top">Upper Top</option>
              <option value="bottoms">Bottoms</option>
              <option value="dress">Dress</option>
            </select>
          </label>
          <label>
            Fit
            <select
              value={sizePayload.fit}
              onChange={(e) => setSizePayload((prev) => ({ ...prev, fit: e.target.value }))}
            >
              <option value="regular">Regular</option>
              <option value="slim">Slim</option>
              <option value="relaxed">Relaxed</option>
            </select>
          </label>
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
          <div>
            <pre>{JSON.stringify(sizeResult, null, 2)}</pre>
            {(sizeResult as any)?.two_size_preview && (
              <button type="button" onClick={handleCompare} disabled={tryOnLoading}>
                {tryOnLoading ? "Comparing…" : "Compare recommended vs nearest"}
              </button>
            )}
          </div>
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
        {compareResult && (
          <div>
            <h3>Two-size preview: {compareResult.sizeA} vs {compareResult.sizeB}</h3>
            <div className="form-row">
              <div style={{ flex: 1 }}>
                <p>
                  {compareResult.sizeA} · Cache: {compareResult.setA.cacheKey} · Conf: {compareResult.setA.confidence.toFixed(2)}
                </p>
                <div className="poses">
                  {compareResult.setA.images.map((img, idx) => (
                    <div className="pose-card" key={img}>
                      <img src={img} alt={`A Pose ${idx + 1}`} />
                      <span>Score: {compareResult.setA.frameScores[idx].toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ flex: 1 }}>
                <p>
                  {compareResult.sizeB} · Cache: {compareResult.setB.cacheKey} · Conf: {compareResult.setB.confidence.toFixed(2)}
                </p>
                <div className="poses">
                  {compareResult.setB.images.map((img, idx) => (
                    <div className="pose-card" key={img}>
                      <img src={img} alt={`B Pose ${idx + 1}`} />
                      <span>Score: {compareResult.setB.frameScores[idx].toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
