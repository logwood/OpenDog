export type GalleryHealth = {
  pets: number;
  reference_images: number;
  experts?: string[];
};

export type HealthResponse = {
  status: string;
  model_fingerprint: string;
  backend: Record<string, unknown>;
  gallery: GalleryHealth;
  operations?: {
    history?: number;
    hard_cases?: number;
    batches?: number;
  };
};

export type PetSummary = {
  pet_id: string;
  display_name: string;
  created_at: string;
  updated_at: string;
  reference_count: number;
};

export type PetImage = {
  image_id: string;
  original_filename: string;
  content_type: string;
  width: number;
  height: number;
  byte_size: number;
  sha256: string;
  quality?: {
    status: "good" | "warning" | "unknown";
    reasons: string[];
    branch_available: boolean[] | null;
    branch_quality: number[] | null;
    architecture?: "unified_single_graph" | string;
  };
  created_at: string;
};

export type PetDetails = PetSummary & {
  images: PetImage[];
};

export type PetListResponse = {
  pets: PetSummary[];
  count: number;
};

export type EnrollmentResponse = {
  pet: PetDetails;
  added_image_ids: string[];
  duplicate_image_ids: string[];
};

export type Candidate = {
  pet_id: string;
  display_name: string;
  score: number;
  reference_count: number;
  expert_scores?: Record<string, number>;
};

export type AgentDecision = {
  decision: "matched" | "needs_more_evidence" | "possible_unknown";
  expert_agreement: boolean;
  calibration: string;
  score_semantics: string;
  expert_weights: Record<string, number>;
  expert_results: Record<string, {
    pet_id: string;
    display_name: string;
    score: number;
    evidence: number;
    margin: number | null;
  }>;
  quality: Record<string, unknown>;
  thresholds: {
    match_score: number;
    minimum_margin: number;
    source: string;
  };
  reasons: string[];
  capture_recommendations: string[];
};

export type DescriptorMetadata = {
  branch_available?: [boolean, boolean] | boolean[];
  fusion_weights?: [number, number] | number[];
  branch_quality?: [number, number] | number[];
  detection?: Record<string, unknown> | null;
  inference_size?: [number, number] | number[] | null;
  runtime_diagnostics?: {
    unified?: {
      single_graph?: boolean;
      external_models?: unknown[];
      provider?: string;
      input_size?: number;
    };
    [key: string]: unknown;
  } | null;
};

export type QueryMetadata = {
  filename?: string;
  sha256?: string;
  width?: number;
  height?: number;
  inference?: {
    detections?: number;
    selected_detection?: number;
    descriptor?: DescriptorMetadata;
  };
};

export type IdentificationResponse = {
  decision: string;
  accepted: boolean;
  predicted_pet_id: string | null;
  predicted_display_name: string | null;
  top1_score: number;
  margin: number | null;
  match_threshold: number | null;
  minimum_margin: number;
  candidates: Candidate[];
  query: QueryMetadata;
  latency_ms?: number;
  model_fingerprint?: string;
  gallery_snapshot?: GalleryHealth;
  diagnostics?: {
    mode?: "unified_single_graph" | "multibranch" | string;
    single_graph?: boolean;
    branch_available?: boolean[] | null;
    branch_quality?: number[] | null;
    branch_top1?: {
      nose?: { pet_id: string; display_name: string; score: number } | null;
      face?: { pet_id: string; display_name: string; score: number } | null;
    };
    branch_conflict?: boolean;
  };
  hard_case_reasons?: string[];
  history_id?: string;
  agent?: AgentDecision;
};

export type ReviewStatus = "unreviewed" | "correct" | "incorrect" | "uncertain";

export type HistoryItem = {
  history_id: string;
  created_at: string;
  source: "single" | "batch" | string;
  batch_id: string | null;
  status: "succeeded" | "failed" | string;
  filename: string;
  sha256: string;
  width: number | null;
  height: number | null;
  byte_size: number;
  image_available: boolean;
  expected_pet_id: string | null;
  accepted: boolean | null;
  predicted_pet_id: string | null;
  predicted_display_name: string | null;
  top1_score: number | null;
  margin: number | null;
  match_threshold: number | null;
  minimum_margin: number | null;
  latency_ms: number | null;
  model_fingerprint: string;
  gallery_snapshot: GalleryHealth;
  review_status: ReviewStatus;
  review_note: string | null;
  reviewed_at: string | null;
  hard_case_reasons: string[];
  error: { code?: string; message?: string } | null;
  result?: IdentificationResponse | null;
};

export type HistoryListResponse = {
  items: HistoryItem[];
  total: number;
  page: number;
  page_size: number;
};

export type BatchMetrics = {
  labelled?: number;
  top1_correct?: number;
  top1_accuracy?: number | null;
  accepted_correct?: number;
  accepted_accuracy?: number | null;
  rejected?: number;
  hard_cases?: number;
  average_latency_ms?: number | null;
  p95_latency_ms?: number | null;
};

export type BatchJob = {
  batch_id: string;
  name: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  total: number;
  completed: number;
  succeeded: number;
  failed: number;
  cancel_requested: boolean;
  model_fingerprint: string;
  parameters: Record<string, unknown>;
  metrics: BatchMetrics;
  error_message: string | null;
  results?: HistoryItem[];
};

export type BatchListResponse = {
  items: BatchJob[];
  total: number;
  page: number;
  page_size: number;
};

type ErrorEnvelope = {
  error?: {
    code?: string;
    message?: string;
    details?: Record<string, unknown>;
  };
};

export class PetReIdApiError extends Error {
  status: number;
  code: string;
  details: Record<string, unknown>;

  constructor(
    message: string,
    status = 0,
    code = "request_failed",
    details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "PetReIdApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

const API_BASE_STORAGE_KEY = "pawprint-id.api-base-url";
const SAME_ORIGIN_SENTINEL = "__same_origin__";

export const DEFAULT_API_BASE_URL = normalizeApiBaseUrl(
  process.env.NEXT_PUBLIC_PET_REID_API_BASE_URL ?? "/",
);

let runtimeApiBaseUrl: string | null = null;

export function normalizeApiBaseUrl(value: string): string {
  const normalized = value.trim();
  if (!normalized || normalized === "/") return "";
  return normalized.replace(/\/+$/, "");
}

function storedApiBaseUrl(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = window.localStorage.getItem(API_BASE_STORAGE_KEY);
    if (stored === SAME_ORIGIN_SENTINEL) return "";
    return stored ? normalizeApiBaseUrl(stored) : null;
  } catch {
    return null;
  }
}

export function getApiBaseUrl(): string {
  if (runtimeApiBaseUrl !== null) return runtimeApiBaseUrl;
  return storedApiBaseUrl() ?? DEFAULT_API_BASE_URL;
}

export function setApiBaseUrl(value: string | null): string {
  if (value === null) {
    runtimeApiBaseUrl = null;
    if (typeof window !== "undefined") {
      try { window.localStorage.removeItem(API_BASE_STORAGE_KEY); } catch { /* in-memory setting still works */ }
    }
    return getApiBaseUrl();
  }

  runtimeApiBaseUrl = normalizeApiBaseUrl(value);
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(
        API_BASE_STORAGE_KEY,
        runtimeApiBaseUrl || SAME_ORIGIN_SENTINEL,
      );
    } catch { /* in-memory setting still works */ }
  }
  return runtimeApiBaseUrl;
}

function apiUrl(path: string): string {
  return getApiBaseUrl() + path;
}

const ERROR_MESSAGES: Record<string, string> = {
  gallery_empty: "临时图库还是空的，请先录入至少一只宠物。",
  invalid_gallery_request: "请求参数不正确，请检查后再试。",
  invalid_request: "请求参数不正确，请检查后再试。",
  invalid_pet_image: "图片无法用于识别，请换一张清晰且只包含一只宠物的照片。",
  pet_not_found: "这只宠物已不存在，图库可能刚刚发生了变化。",
  not_found: "记录不存在或已被删除。",
  image_not_found: "这张参考图已不存在。",
  gallery_conflict: "这张图片已经属于另一只宠物，不能重复录入。",
  gallery_model_mismatch: "图库与当前模型不兼容，需要重新建立临时图库。",
  model_mismatch: "图库与当前模型不兼容，需要重新建立临时图库。",
  upstream_unavailable: "Java 网关无法连接推理服务。",
  request_too_large: "图片或本次上传内容过大。",
  upload_too_large: "图片或本次上传内容过大。",
  method_argument_not_valid: "填写内容不符合接口要求。",
  constraint_violation: "填写内容不符合接口要求。",
  admin_unauthorized: "管理员密钥无效或当前服务没有配置管理员密钥。",
};

export function errorMessage(error: unknown): string {
  if (error instanceof PetReIdApiError) {
    return ERROR_MESSAGES[error.code] ??
      (error.status === 413
        ? "图片或本次上传内容过大。"
        : error.status === 502
          ? "Java 网关无法连接推理服务。"
          : "请求没有完成，请稍后重试。（" + error.code + "）");
  }
  if (error instanceof TypeError) {
    const configured = getApiBaseUrl();
    return configured
      ? "无法连接服务端 " + configured + "，请检查地址、网络和跨域设置。"
      : "无法通过当前网页的 /v1 代理连接识别服务，请确认电脑端完整服务仍在运行。";
  }
  return "发生了意外错误，请稍后重试。";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), {
      ...init,
      headers: {
        Accept: "application/json",
        ...init?.headers,
      },
    });
  } catch (error) {
    throw error instanceof TypeError
      ? error
      : new PetReIdApiError("Network request failed");
  }

  if (!response.ok) {
    let envelope: ErrorEnvelope = {};
    try {
      envelope = (await response.json()) as ErrorEnvelope;
    } catch {
      // Non-JSON errors are normalized below.
    }
    throw new PetReIdApiError(
      envelope.error?.message ?? "Request failed",
      response.status,
      envelope.error?.code ?? "http_" + response.status,
      envelope.error?.details ?? {},
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

async function download(path: string, adminKey?: string): Promise<Blob> {
  const response = await fetch(apiUrl(path), {
    headers: {
      Accept: "*/*",
      ...(adminKey ? { "X-Admin-Key": adminKey } : {}),
    },
  });
  if (!response.ok) {
    let envelope: ErrorEnvelope = {};
    try { envelope = (await response.json()) as ErrorEnvelope; } catch { /* normalized below */ }
    throw new PetReIdApiError(
      envelope.error?.message ?? "Download failed",
      response.status,
      envelope.error?.code ?? "http_" + response.status,
      envelope.error?.details ?? {},
    );
  }
  return response.blob();
}

function adminHeaders(adminKey: string): HeadersInit {
  return { "X-Admin-Key": adminKey };
}

function historyQuery(filters: {
  page?: number;
  pageSize?: number;
  source?: string;
  accepted?: boolean;
  reviewStatus?: string;
  petId?: string;
} = {}): string {
  const query = new URLSearchParams({
    page: String(filters.page ?? 1),
    page_size: String(filters.pageSize ?? 25),
  });
  if (filters.source) query.set("source", filters.source);
  if (typeof filters.accepted === "boolean") query.set("accepted", String(filters.accepted));
  if (filters.reviewStatus) query.set("review_status", filters.reviewStatus);
  if (filters.petId) query.set("pet_id", filters.petId);
  return query.toString();
}

export const petReIdApi = {
  health: () => request<HealthResponse>("/v1/upstream-health"),

  listPets: () => request<PetListResponse>("/v1/pets"),

  getPet: (petId: string) =>
    request<PetDetails>("/v1/pets/" + encodeURIComponent(petId)),

  updatePet: (petId: string, displayName: string) =>
    request<PetDetails>("/v1/pets/" + encodeURIComponent(petId), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName }),
    }),

  identify: (file: File, topK = 5) => {
    const form = new FormData();
    form.append("file", file);
    return request<IdentificationResponse>("/v1/identify?top_k=" + topK, {
      method: "POST",
      body: form,
    });
  },

  enroll: (petId: string, displayName: string, files: File[]) => {
    const form = new FormData();
    if (displayName.trim()) {
      form.append("display_name", displayName.trim());
    }
    files.forEach((file) => form.append("files", file));
    return request<EnrollmentResponse>(
      "/v1/pets/" + encodeURIComponent(petId) + "/images",
      { method: "POST", body: form },
    );
  },

  deleteImage: (petId: string, imageId: string) =>
    request<{ pet_id: string; deleted_image_id: string; remaining_references: number; pet_deleted: boolean }>(
      "/v1/pets/" + encodeURIComponent(petId) + "/images/" + encodeURIComponent(imageId),
      { method: "DELETE" },
    ),

  deletePet: (petId: string) =>
    request<{ deleted_pet_id: string; deleted_images: number }>(
      "/v1/pets/" + encodeURIComponent(petId),
      { method: "DELETE" },
    ),

  listHistory: (filters?: Parameters<typeof historyQuery>[0]) =>
    request<HistoryListResponse>("/v1/history?" + historyQuery(filters)),

  getHistory: (historyId: string) =>
    request<HistoryItem>("/v1/history/" + encodeURIComponent(historyId)),

  reviewHistory: (historyId: string, status: ReviewStatus, note = "") =>
    request<HistoryItem>("/v1/history/" + encodeURIComponent(historyId) + "/review", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, note }),
    }),

  deleteHistory: (historyId: string) =>
    request<{ deleted_history_id: string }>("/v1/history/" + encodeURIComponent(historyId), {
      method: "DELETE",
    }),

  historyImageUrl: (historyId: string) =>
    apiUrl("/v1/history/" + encodeURIComponent(historyId) + "/image"),

  adminAccess: (adminKey: string) =>
    request<{ authorized: boolean }>("/v1/admin/access", { headers: adminHeaders(adminKey) }),

  createBatch: (
    adminKey: string,
    name: string,
    files: File[],
    expectedPetIds: (string | null)[],
    topK = 5,
  ) => {
    const form = new FormData();
    form.append("name", name.trim() || "批量测试");
    files.forEach((file) => form.append("files", file));
    if (expectedPetIds.some(Boolean)) {
      expectedPetIds.forEach((petId) => form.append("expected_pet_ids", petId ?? ""));
    }
    return request<BatchJob>("/v1/admin/batches?top_k=" + topK, {
      method: "POST",
      headers: adminHeaders(adminKey),
      body: form,
    });
  },

  listBatches: (adminKey: string) =>
    request<BatchListResponse>("/v1/admin/batches?page=1&page_size=50", {
      headers: adminHeaders(adminKey),
    }),

  getBatch: (adminKey: string, batchId: string) =>
    request<BatchJob>("/v1/admin/batches/" + encodeURIComponent(batchId), {
      headers: adminHeaders(adminKey),
    }),

  cancelBatch: (adminKey: string, batchId: string) =>
    request<BatchJob>("/v1/admin/batches/" + encodeURIComponent(batchId), {
      method: "DELETE",
      headers: adminHeaders(adminKey),
    }),

  listHardCases: (adminKey: string) =>
    request<HistoryListResponse>("/v1/admin/hard-cases?page=1&page_size=100", {
      headers: adminHeaders(adminKey),
    }),

  downloadBatchCsv: (adminKey: string, batchId: string) =>
    download("/v1/admin/batches/" + encodeURIComponent(batchId) + "/results.csv", adminKey),

  downloadGalleryBackup: (adminKey: string) =>
    download("/v1/admin/gallery/backup", adminKey),

  restoreGallery: (adminKey: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ pets: number; added_images: number; duplicate_images: number; mode: string }>(
      "/v1/admin/gallery/restore",
      { method: "POST", headers: adminHeaders(adminKey), body: form },
    );
  },

  imageUrl: (petId: string, imageId: string) =>
    getApiBaseUrl() +
    "/v1/pets/" +
    encodeURIComponent(petId) +
    "/images/" +
    encodeURIComponent(imageId),
};

export function getDescriptor(result: IdentificationResponse | null): DescriptorMetadata | null {
  return result?.query?.inference?.descriptor ?? null;
}

export function isUnifiedDescriptor(descriptor: DescriptorMetadata | null): boolean {
  return descriptor?.runtime_diagnostics?.unified?.single_graph === true;
}

export function fusionState(
  descriptor: DescriptorMetadata | null,
): "unified" | "joint" | "fallback" | "unknown" {
  if (isUnifiedDescriptor(descriptor)) return "unified";
  const available = descriptor?.branch_available;
  if (!Array.isArray(available) || available.length < 2) return "unknown";
  return available[0] && available[1] ? "joint" : "fallback";
}
