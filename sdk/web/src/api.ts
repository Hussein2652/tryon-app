import axios from "axios";

export interface SizeInputs {
  method: string;
  chest_circ_cm?: number;
  waist_circ_cm?: number;
  bust_circ_cm?: number;
}

export interface SizeRecommendPayload {
  brand_id: string;
  category: string;
  fit: string;
  stretch_pct: number;
  inputs: SizeInputs;
  size_chart: Array<{ size: string; [key: string]: number | string }>;
}

export async function recommendSize(payload: SizeRecommendPayload) {
  const response = await axios.post("/size/recommend", payload);
  return response.data;
}

export interface TryOnResponse {
  ok: boolean;
  cache_key: string;
  images: string[];
  frame_scores: number[];
  confidence_avg: number;
}

export async function requestTryOn(formData: FormData): Promise<TryOnResponse> {
  const response = await axios.post("/tryon/preview", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return response.data;
}

export async function deleteTryOn(cacheKey: string) {
  const response = await axios.delete(`/tryon/${cacheKey}`);
  return response.data;
}
