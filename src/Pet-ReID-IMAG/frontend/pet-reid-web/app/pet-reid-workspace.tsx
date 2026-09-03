"use client";

/* eslint-disable @next/next/no-img-element */
import {
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type EnrollmentResponse,
  type HealthResponse,
  type IdentificationResponse,
  type PetDetails,
  type PetSummary,
  errorMessage,
  fusionState,
  getDescriptor,
  petReIdApi,
} from "../lib/pet-reid-api";
import MobileControls from "./mobile-controls";
import { AdminSection, HistorySection } from "./workspace-sections";

const MAX_IMAGE_BYTES = 15 * 1024 * 1024;
const IMAGE_EXTENSIONS = /\.(jpe?g|png|webp|bmp)$/i;
const PET_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
type ServiceState = "checking" | "online" | "offline";

function validateImage(file: File): string | null {
  const supported = ["image/jpeg", "image/png", "image/webp", "image/bmp"].includes(file.type);
  if (!supported && !IMAGE_EXTENSIONS.test(file.name)) return "请选择 JPG、PNG、WebP 或 BMP 图片。";
  if (file.size > MAX_IMAGE_BYTES) return "单张图片不能超过 15 MB。";
  if (!file.size) return "图片文件是空的，请重新选择。";
  return null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "—";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function score(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
}

function backendValue(health: HealthResponse | null, key: string): string {
  const value = health?.backend?.[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : "—";
}

function modelLabel(health: HealthResponse | null): string {
  const displayName = backendValue(health, "display_name");
  if (displayName !== "—") return displayName;
  const role = backendValue(health, "deployment_role");
  if (role === "candidate") return "高分辨率统一识别 · 当前候选";
  if (role === "production") return "统一识别 · 生产基线";
  if (role === "rollback") return "统一识别 · 回滚点";
  const capability = backendValue(health, "capability");
  if (capability === "multi-expert-evidence") return "多专家证据融合 · 研究实验";
  if (capability === "semantic-body-fusion") return "语义与身体融合 · 研究实验";
  if (capability === "semantic-compatibility") return "语义识别 · 兼容模式";
  if (health?.backend?.single_graph === true) return "统一识别 · 单图模型";
  return "多分支兼容模型";
}

function modelSummary(health: HealthResponse | null): string {
  const summary = backendValue(health, "summary");
  if (summary !== "—") return summary;
  if (
    health?.backend?.backend === "onnxruntime-unified" ||
    health?.backend?.single_graph === true
  ) return "单一联合模型 · RGB → 512D";
  const dimension = backendValue(health, "embedding_dim");
  return (dimension === "—" ? "512" : dimension) + " 维特征";
}

function providerLabel(health: HealthResponse | null): string {
  const provider = backendValue(health, "provider");
  if (provider.includes("CUDA")) return "CUDA ONNX";
  if (provider === "—") return "等待连接";
  return provider.replace("ExecutionProvider", "");
}

function qualityReason(reason: string): string {
  return ({
    single_branch: "单分支",
    low_nose_quality: "鼻部质量低",
    low_face_quality: "脸部质量低",
    diagnostics_unavailable: "无质量诊断",
  } as Record<string, string>)[reason] ?? reason;
}

function ResultPanel({ result, preview, onReset }: {
  result: IdentificationResponse;
  preview: string | null;
  onReset: () => void;
}) {
  const descriptor = getDescriptor(result);
  const fusion = fusionState(descriptor);
  const unified = fusion === "unified" || result.diagnostics?.single_graph === true;
  const weights = descriptor?.fusion_weights;
  const noseWeight = Array.isArray(weights) && typeof weights[0] === "number" ? weights[0] : null;
  const faceWeight = Array.isArray(weights) && typeof weights[1] === "number" ? weights[1] : null;
  const identity = result.accepted
    ? result.predicted_display_name || result.predicted_pet_id
    : "未确认身份";
  const agent = result.agent;
  const verdict = agent?.decision === "possible_unknown"
    ? "可能是库外身份"
    : agent?.decision === "needs_more_evidence"
      ? "需要更多证据"
      : result.accepted ? "已匹配" : "未通过阈值";

  return (
    <section className="result-panel panel" aria-live="polite">
      <div className="result-visual">
        {preview ? <img src={preview} alt="本次比对的宠物照片" /> : null}
        <span className={"result-verdict " + (result.accepted ? "accepted" : "rejected")}>
          {verdict}
        </span>
      </div>
      <div className="result-main">
        <div className="result-heading">
          <div>
            <p className="eyebrow">识别结果</p>
            <h2>{identity}</h2>
            {result.accepted ? <p className="result-pet-id">身份 ID · {result.predicted_pet_id}</p> : null}
          </div>
          <button className="text-button" type="button" onClick={onReset}>换一张照片</button>
        </div>

        <div className="score-grid">
          <div><span>Top-1 分数</span><strong>{score(result.top1_score)}</strong><small>不是概率</small></div>
          <div><span>Top-1 / Top-2 差值</span><strong>{score(result.margin)}</strong><small>越大区分越明显</small></div>
          <div>
            <span>{unified ? "模型路径" : "特征分支"}</span>
            <strong className={unified || fusion === "joint" ? "good-text" : fusion === "fallback" ? "warn-text" : ""}>
              {unified ? "RGB → 512D" : fusion === "joint" ? "鼻子 + 脸" : fusion === "fallback" ? "单路回退" : "未知"}
            </strong>
            <small>{unified ? "单一联合模型" : fusion === "joint" ? "双分支" : fusion === "fallback" ? "单分支" : "无分支信息"}</small>
          </div>
        </div>

        {result.decision === "closed_set_top1" ? (
          <div className="notice warning"><span>!</span><p><strong>未启用拒识</strong>结果只表示图库中最相似的身份，不代表身份概率。</p></div>
        ) : null}
        {!unified && fusion === "fallback" ? (
          <div className="notice warning"><span>!</span><p><strong>单分支结果</strong>没有同时取得鼻子和脸部特征，建议更换图片。</p></div>
        ) : null}
        {agent && !agent.expert_agreement ? (
          <div className="notice warning"><span>!</span><p><strong>专家意见不一致</strong>BIFOR 与身形专家指向不同身份，本次不做硬判。</p></div>
        ) : null}
        {agent?.capture_recommendations.map((recommendation) => (
          <div className="notice warning" key={recommendation}><span>↻</span><p>{recommendation}</p></div>
        ))}

        {agent ? (
          <div className="fusion-readout">
            <div className="fusion-title">
              <span>专家证据权重</span>
              <small>无训练单调融合 · 不是概率</small>
            </div>
            {Object.entries(agent.expert_weights).map(([expertId, weight]) => (
              <div className={"weight-row " + (expertId === "bifor" ? "" : "face")} key={expertId}>
                <span>{expertId === "bifor" ? "BIFOR 鼻脸身体" : "MegaDescriptor 身形"}</span>
                <div><i style={{ width: Math.max(2, weight * 100) + "%" }} /></div>
                <strong>{(weight * 100).toFixed(1)}%</strong>
              </div>
            ))}
          </div>
        ) : null}

        {!unified ? <div className="fusion-readout">
          <div className="fusion-title">
            <span>分支权重</span>
            <small>{result.query.width ?? "—"} × {result.query.height ?? "—"} px · 检测到 {result.query.inference?.detections ?? "—"} 只</small>
          </div>
          <div className="weight-row">
            <span>鼻子分支</span><div><i style={{ width: noseWeight === null ? "0%" : Math.max(2, noseWeight * 100) + "%" }} /></div>
            <strong>{noseWeight === null ? "—" : (noseWeight * 100).toFixed(1) + "%"}</strong>
          </div>
          <div className="weight-row face">
            <span>脸部分支</span><div><i style={{ width: faceWeight === null ? "0%" : Math.max(2, faceWeight * 100) + "%" }} /></div>
            <strong>{faceWeight === null ? "—" : (faceWeight * 100).toFixed(1) + "%"}</strong>
          </div>
        </div> : null}

        <div className="candidate-list">
          <div className="candidate-header"><span>候选身份</span><span>相似度</span></div>
          {result.candidates.map((candidate, index) => (
            <div className="candidate-row" key={candidate.pet_id}>
              <span className="candidate-rank">{String(index + 1).padStart(2, "0")}</span>
              <div>
                <strong>{candidate.display_name || candidate.pet_id}</strong>
                <small>{candidate.pet_id} · {candidate.reference_count} 张参考图
                  {candidate.expert_scores ? " · BIFOR " + score(candidate.expert_scores.bifor) + " / 身形 " + score(candidate.expert_scores.megadescriptor_b224) : ""}
                </small>
              </div>
              <div className="candidate-score"><strong>{score(candidate.score)}</strong><span><i style={{ width: Math.max(3, Math.min(100, Math.max(0, candidate.score) * 100)) + "%" }} /></span></div>
            </div>
          ))}
          {!result.candidates.length ? <p className="empty-inline">服务没有返回候选身份。</p> : null}
        </div>
      </div>
    </section>
  );
}

function EnrollmentDialog({ open, onClose, onComplete }: {
  open: boolean;
  onClose: () => void;
  onComplete: (result: EnrollmentResponse) => Promise<void>;
}) {
  const [petId, setPetId] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);

  const reset = useCallback(() => {
    setPetId(""); setDisplayName(""); setFiles([]); setError(null);
  }, []);
  const close = () => { if (!submitting) { reset(); onClose(); } };

  const addFiles = (incoming: File[]) => {
    setError(null);
    for (const file of incoming) {
      const validation = validateImage(file);
      if (validation) { setError(file.name + "：" + validation); return; }
    }
    const unique = [...files];
    for (const file of incoming) {
      if (!unique.some((item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified)) unique.push(file);
    }
    if (unique.length > 8) { setError("一次最多录入 8 张参考图。请删减后重试。"); return; }
    setFiles(unique);
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalizedId = petId.trim();
    if (!PET_ID_PATTERN.test(normalizedId)) {
      setError("身份 ID 需以字母或数字开头，只能包含字母、数字、点、下划线和短横线，最长 64 位。"); return;
    }
    if (displayName.trim().length > 128) { setError("显示名称不能超过 128 个字符。"); return; }
    if (files.length < 1 || files.length > 8) { setError("请选择 1–8 张参考图。"); return; }
    setSubmitting(true); setError(null);
    try {
      const response = await petReIdApi.enroll(normalizedId, displayName, files);
      await onComplete(response); reset(); onClose();
    } catch (requestError) { setError(errorMessage(requestError)); }
    finally { setSubmitting(false); }
  };

  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && close()}>
      <section className="dialog-card" role="dialog" aria-modal="true" aria-labelledby="enroll-title">
        <div className="dialog-heading">
          <div><h2 id="enroll-title">录入宠物</h2></div>
          <button className="icon-button" type="button" onClick={close} aria-label="关闭录入窗口">×</button>
        </div>
        <p className="dialog-copy">每张图片只能包含一只宠物，建议录入 2–5 张。</p>
        <form onSubmit={submit}>
          <div className="form-grid">
            <label><span>身份 ID <b>必填</b></span><input autoFocus value={petId} onChange={(event) => setPetId(event.target.value)} placeholder="例如 dog-001" maxLength={64} /></label>
            <label><span>显示名称 <small>可选</small></span><input value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="例如 豆豆" maxLength={128} /></label>
          </div>
          <div className="enrollment-image-actions">
            <button className="mini-upload" type="button" onClick={() => inputRef.current?.click()}><span>+</span><strong>{files.length ? "继续添加参考图" : "选择 1–8 张参考图"}</strong><small>相册或本地文件</small></button>
            <button className="mini-upload camera-upload" type="button" onClick={() => cameraRef.current?.click()}><span>◎</span><strong>拍摄参考图</strong><small>使用手机后置相机</small></button>
          </div>
          <input ref={inputRef} className="sr-only" type="file" multiple accept="image/jpeg,image/png,image/webp,image/bmp" onChange={(event: ChangeEvent<HTMLInputElement>) => { addFiles(Array.from(event.target.files ?? [])); event.target.value = ""; }} />
          <input ref={cameraRef} className="sr-only" type="file" accept="image/*" capture="environment" onChange={(event: ChangeEvent<HTMLInputElement>) => { addFiles(Array.from(event.target.files ?? [])); event.target.value = ""; }} />
          {files.length ? (
            <div className="selected-files">
              {files.map((file, index) => (
                <div key={file.name + file.lastModified}>
                  <span>{String(index + 1).padStart(2, "0")}</span><p><strong>{file.name}</strong><small>{formatBytes(file.size)}</small></p>
                  <button type="button" onClick={() => setFiles(files.filter((item) => item !== file))} aria-label={"移除 " + file.name}>×</button>
                </div>
              ))}
            </div>
          ) : null}
          {error ? <p className="form-error" role="alert">{error}</p> : null}
          <div className="dialog-actions">
            <button className="secondary-button" type="button" onClick={close} disabled={submitting}>取消</button>
            <button className="primary-button" type="submit" disabled={submitting || !files.length}>{submitting ? "处理中…" : "保存"}</button>
          </div>
        </form>
      </section>
    </div>
  );
}

function PetDetailDrawer({ petId, pet, loading, error, onClose, onReload, onChanged }: {
  petId: string | null;
  pet: PetDetails | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onReload: () => Promise<void>;
  onChanged: (message: string) => Promise<void>;
}) {
  const [busyId, setBusyId] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState("");
  const addImageRef = useRef<HTMLInputElement>(null);
  const addImageCameraRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const timer = window.setTimeout(() => setDisplayName(pet?.display_name ?? ""), 0);
    return () => window.clearTimeout(timer);
  }, [pet?.display_name]);
  if (!petId) return null;

  const saveName = async () => {
    const value = displayName.trim();
    if (!value) { window.alert("显示名称不能为空。"); return; }
    setBusyId("__name__");
    try {
      await petReIdApi.updatePet(petId, value);
      await onChanged("显示名称已更新。");
      await onReload();
    } catch (requestError) { window.alert(errorMessage(requestError)); }
    finally { setBusyId(null); }
  };

  const addImages = async (incoming: File[]) => {
    if (!incoming.length) return;
    for (const file of incoming) {
      const validation = validateImage(file);
      if (validation) { window.alert(file.name + "：" + validation); return; }
    }
    if (incoming.length > 8) { window.alert("一次最多补充 8 张参考图。"); return; }
    setBusyId("__add__");
    try {
      const response = await petReIdApi.enroll(petId, pet?.display_name ?? "", incoming);
      await onChanged("已新增 " + response.added_image_ids.length + " 张参考图" + (response.duplicate_image_ids.length ? "，跳过重复 " + response.duplicate_image_ids.length + " 张。" : "。"));
      await onReload();
    } catch (requestError) { window.alert(errorMessage(requestError)); }
    finally {
      setBusyId(null);
      if (addImageRef.current) addImageRef.current.value = "";
      if (addImageCameraRef.current) addImageCameraRef.current.value = "";
    }
  };

  const deleteImage = async (imageId: string, filename: string) => {
    if (!window.confirm("确认从临时图库删除参考图“" + filename + "”吗？此操作不可撤销。")) return;
    setBusyId(imageId);
    try {
      const response = await petReIdApi.deleteImage(petId, imageId);
      if (response.pet_deleted) { await onChanged("最后一张参考图已删除，宠物身份也已移除。"); onClose(); }
      else { await onChanged("参考图已从临时图库删除。"); await onReload(); }
    } catch (requestError) { window.alert(errorMessage(requestError)); }
    finally { setBusyId(null); }
  };

  const deletePet = async () => {
    if (!window.confirm("确认删除“" + (pet?.display_name || petId) + "”及其全部参考图吗？此操作不可撤销。")) return;
    setBusyId("__pet__");
    try { await petReIdApi.deletePet(petId); await onChanged("宠物身份及其参考图已删除。"); onClose(); }
    catch (requestError) { window.alert(errorMessage(requestError)); }
    finally { setBusyId(null); }
  };

  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <aside className="pet-drawer" role="dialog" aria-modal="true" aria-labelledby="pet-detail-title">
        <div className="drawer-heading">
          <div><h2 id="pet-detail-title">{pet?.display_name || petId}</h2><small>{petId}</small></div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="关闭宠物详情">×</button>
        </div>
        {loading ? <div className="drawer-loading"><span /><p>读取参考图…</p></div> : null}
        {error ? <div className="drawer-error" role="alert"><p>{error}</p><button type="button" onClick={() => void onReload()}>重试</button></div> : null}
        {pet && !loading ? (
          <>
            <div className="pet-edit-row">
              <label><span>显示名称</span><input value={displayName} maxLength={128} onChange={(event) => setDisplayName(event.target.value)} /></label>
              <button type="button" disabled={busyId === "__name__" || displayName.trim() === pet.display_name} onClick={() => void saveName()}>{busyId === "__name__" ? "保存中" : "保存"}</button>
            </div>
            <div className="detail-meta"><div><strong>{pet.reference_count}</strong><span>参考图片</span></div><div><strong>{formatDate(pet.updated_at)}</strong><span>最近更新</span></div></div>
            <div className="drawer-toolbar"><button type="button" disabled={busyId === "__add__"} onClick={() => addImageCameraRef.current?.click()}>拍照补充</button><button type="button" disabled={busyId === "__add__"} onClick={() => addImageRef.current?.click()}>{busyId === "__add__" ? "处理中…" : "+ 选择参考图"}</button><input ref={addImageCameraRef} className="sr-only" type="file" accept="image/*" capture="environment" onChange={(event) => void addImages(Array.from(event.target.files ?? []))} /><input ref={addImageRef} className="sr-only" type="file" multiple accept="image/jpeg,image/png,image/webp,image/bmp" onChange={(event) => void addImages(Array.from(event.target.files ?? []))} /></div>
            <div className="reference-grid">
              {pet.images.map((image) => (
                <article className="reference-card" key={image.image_id}>
                  <div className="reference-image"><img src={petReIdApi.imageUrl(petId, image.image_id)} alt={image.original_filename} loading="lazy" /></div>
                  <div className="reference-info"><strong title={image.original_filename}>{image.original_filename}</strong><small>{image.width} × {image.height} · {formatBytes(image.byte_size)}</small><span className={"quality-chip " + (image.quality?.status ?? "unknown")}>{image.quality?.status === "good" ? "质量正常" : image.quality?.reasons?.map(qualityReason).join(" / ") || "质量未知"}</span></div>
                  <div className="reference-actions"><a href={petReIdApi.imageUrl(petId, image.image_id)} target="_blank" rel="noreferrer">查看</a><button type="button" disabled={busyId === image.image_id} onClick={() => void deleteImage(image.image_id, image.original_filename)}>{busyId === image.image_id ? "删除中" : "删除"}</button></div>
                </article>
              ))}
            </div>
            <div className="danger-zone"><div><strong>移除整个身份</strong><p>会同时删除全部参考图和已提取的特征。</p></div><button type="button" onClick={() => void deletePet()} disabled={busyId === "__pet__"}>{busyId === "__pet__" ? "删除中…" : "删除身份"}</button></div>
          </>
        ) : null}
      </aside>
    </div>
  );
}

export default function PetReIdWorkspace() {
  const [service, setService] = useState<ServiceState>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [pets, setPets] = useState<PetSummary[]>([]);
  const [refreshing, setRefreshing] = useState(true);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [identifying, setIdentifying] = useState(false);
  const [result, setResult] = useState<IdentificationResponse | null>(null);
  const [dragging, setDragging] = useState(false);
  const [enrollOpen, setEnrollOpen] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [detailPetId, setDetailPetId] = useState<string | null>(null);
  const [detailPet, setDetailPet] = useState<PetDetails | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0);
  const [endpointRevision, setEndpointRevision] = useState(0);
  const queryInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const queryPreview = useMemo(() => queryFile ? URL.createObjectURL(queryFile) : null, [queryFile]);
  useEffect(() => () => { if (queryPreview) URL.revokeObjectURL(queryPreview); }, [queryPreview]);

  const refreshWorkspace = useCallback(async (silent = false) => {
    if (!silent) { setRefreshing(true); setService("checking"); }
    try {
      const [nextHealth, gallery] = await Promise.all([petReIdApi.health(), petReIdApi.listPets()]);
      setHealth(nextHealth); setPets(gallery.pets); setService("online"); setConnectionError(null);
    } catch (requestError) { setService("offline"); setConnectionError(errorMessage(requestError)); }
    finally { setRefreshing(false); }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refreshWorkspace(), 0);
    const timer = window.setInterval(() => void refreshWorkspace(true), 30000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refreshWorkspace]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") { setEnrollOpen(false); setDetailPetId(null); } };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 5000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const chooseQuery = (file: File | undefined) => {
    if (!file) return;
    const validation = validateImage(file);
    if (validation) { setQueryError(validation); return; }
    setQueryFile(file); setResult(null); setQueryError(null);
  };

  const resetQuery = () => {
    setQueryFile(null); setResult(null); setQueryError(null);
    if (queryInputRef.current) queryInputRef.current.value = "";
    if (cameraInputRef.current) cameraInputRef.current.value = "";
  };

  const identify = async () => {
    if (!queryFile || identifying) return;
    setIdentifying(true); setQueryError(null); setResult(null);
    try {
      setResult(await petReIdApi.identify(queryFile, 5));
      setHistoryRefreshToken((value) => value + 1);
      setService("online");
    }
    catch (requestError) { setQueryError(errorMessage(requestError)); if (requestError instanceof TypeError) setService("offline"); }
    finally { setIdentifying(false); }
  };

  const loadPetDetails = useCallback(async (petId: string) => {
    setDetailLoading(true); setDetailError(null);
    try { setDetailPet(await petReIdApi.getPet(petId)); }
    catch (requestError) { setDetailPet(null); setDetailError(errorMessage(requestError)); }
    finally { setDetailLoading(false); }
  }, []);

  const openPet = (petId: string) => { setDetailPetId(petId); setDetailPet(null); void loadPetDetails(petId); };
  const onEnrollmentComplete = async (enrollment: EnrollmentResponse) => {
    const added = enrollment.added_image_ids.length;
    const duplicates = enrollment.duplicate_image_ids.length;
    setToast("已录入 “" + enrollment.pet.display_name + "”：新增 " + added + " 张" + (duplicates ? "，跳过重复 " + duplicates + " 张。" : "。"));
    await refreshWorkspace(true);
  };
  const onGalleryChanged = async (message: string) => { setToast(message); await refreshWorkspace(true); };
  const onEndpointChanged = async () => {
    setResult(null);
    setDetailPetId(null);
    setDetailPet(null);
    setHistoryRefreshToken((value) => value + 1);
    setEndpointRevision((value) => value + 1);
    await refreshWorkspace();
  };
  const referenceCount = health?.gallery.reference_images ?? pets.reduce((sum, pet) => sum + pet.reference_count, 0);
  const fingerprint = health?.model_fingerprint ? health.model_fingerprint.slice(0, 12) : "—";

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#identify" aria-label="Pawprint ID 首页"><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span><span><strong>Pawprint ID</strong><small>本地识别</small></span></a>
        <div className="topbar-actions">
          <button className={"service-pill " + service} type="button" onClick={() => void refreshWorkspace()} disabled={refreshing}><span className="service-dot" />{service === "online" ? "已连接" : service === "checking" ? "连接中" : "未连接"}</button>
          <MobileControls onEndpointChanged={onEndpointChanged} />
        </div>
      </header>

      <div className="workspace">
        <nav className="workspace-nav" aria-label="主要功能">
          <a className="nav-item active" href="#identify">图片比对</a>
          <a className="nav-item" href="#history">比对历史</a>
          <a className="nav-item" href="#gallery">图库管理</a>
          <a className="nav-item" href="#admin">管理员</a>
          <a className="nav-item" href="#system">系统状态</a>
        </nav>

        <section className="content">
          <div className="intro"><div><h1>宠物图片比对</h1></div><p className="intro-copy">拍照或选择一张图片，与临时图库中的身份进行比较。</p></div>

          {connectionError ? <div className="connection-banner" role="alert"><div><span>×</span><p><strong>服务未连接</strong>{connectionError}</p></div><button type="button" onClick={() => void refreshWorkspace()} disabled={refreshing}>{refreshing ? "连接中…" : "重试"}</button></div> : null}

          <div className="dashboard-grid" id="identify">
            <section className="panel identify-panel">
              <div className="panel-heading"><div><h2>选择图片</h2></div><span className="privacy-note">私有服务</span></div>
              <label className={"upload-zone " + (queryFile ? "has-file " : "") + (dragging ? "dragging" : "")} onDragEnter={(event: DragEvent<HTMLLabelElement>) => { event.preventDefault(); setDragging(true); }} onDragOver={(event: DragEvent<HTMLLabelElement>) => event.preventDefault()} onDragLeave={() => setDragging(false)} onDrop={(event: DragEvent<HTMLLabelElement>) => { event.preventDefault(); setDragging(false); chooseQuery(event.dataTransfer.files[0]); }}>
                <input ref={queryInputRef} type="file" accept="image/jpeg,image/png,image/webp,image/bmp" onChange={(event) => chooseQuery(event.target.files?.[0])} />
                {queryPreview ? <div className="query-preview"><img src={queryPreview} alt="待比对宠物照片预览" /><span>点击可更换照片</span></div> : <><span className="upload-orbit" aria-hidden="true"><span>+</span></span><strong>{dragging ? "松开即可选择" : "拖入照片，或点击选择"}</strong><small>手机可直接拍照 · 支持 JPG、PNG、WebP、BMP</small></>}
              </label>
              <input ref={cameraInputRef} className="sr-only" type="file" accept="image/*" capture="environment" onChange={(event) => { chooseQuery(event.target.files?.[0]); event.target.value = ""; }} />
              {queryFile ? <p className="chosen-file"><span>{queryFile.name}</span><small>{formatBytes(queryFile.size)}</small></p> : null}
              {queryError ? <p className="form-error identify-error" role="alert">{queryError}</p> : null}
              <div className="upload-footer"><p><span />图片中只保留一只宠物</p><div className="upload-actions"><button className="camera-button" type="button" disabled={identifying} onClick={() => cameraInputRef.current?.click()}>拍照</button><button type="button" disabled={!queryFile || identifying} onClick={() => void identify()}>{identifying ? <><i className="spinner" />比对中…</> : queryFile ? "开始比对" : "请选择图片"}</button></div></div>
            </section>

            <aside className="summary-stack" id="system">
              <section className="panel model-card"><span className="card-index">模型</span><h2>{modelLabel(health)}</h2><p>{modelSummary(health)}</p><dl><div><dt>后端</dt><dd>{providerLabel(health)}</dd></div><div><dt>指纹</dt><dd title={health?.model_fingerprint}>{fingerprint}</dd></div></dl></section>
              <section className="panel count-card"><span className="card-index">图库</span><div className="count-row"><div><strong>{health?.gallery.pets ?? pets.length}</strong><span>身份</span></div><div><strong>{referenceCount}</strong><span>参考图</span></div></div><p>{service === "online" ? "已同步" : "未连接"}</p></section>
            </aside>
          </div>

          {result ? <ResultPanel result={result} preview={queryPreview} onReset={resetQuery} /> : null}

          <HistorySection refreshToken={historyRefreshToken} />

          <section className="gallery-preview" id="gallery">
            <div className="section-title"><div><h2>临时图库</h2></div><div className="section-actions"><button className="refresh-button" type="button" onClick={() => void refreshWorkspace()} disabled={refreshing} aria-label="刷新临时图库">↻</button><button type="button" onClick={() => setEnrollOpen(true)}>+ 录入</button></div></div>
            <div className="pet-list">
              {pets.map((pet, index) => (
                <article className="pet-row" key={pet.pet_id}><span className={"pet-avatar " + (index % 2 ? "peach" : "mint")}>{(pet.display_name || pet.pet_id).trim().slice(0, 1).toUpperCase()}</span><div className="pet-identity"><strong>{pet.display_name || pet.pet_id}</strong><small>{pet.pet_id} · {formatDate(pet.updated_at)}</small></div><span className="reference-count">{pet.reference_count} 张参考图</span><button type="button" onClick={() => openPet(pet.pet_id)} aria-label={"查看 " + (pet.display_name || pet.pet_id)}>→</button></article>
              ))}
              {!refreshing && !pets.length ? <div className="empty-gallery"><span>+</span><div><strong>图库为空</strong><p>请先录入参考图片。</p></div><button type="button" onClick={() => setEnrollOpen(true)}>录入</button></div> : null}
              {refreshing && !pets.length ? <div className="gallery-skeleton"><i /><i /></div> : null}
            </div>
          </section>

          <AdminSection key={endpointRevision} onGalleryChanged={onGalleryChanged} />
        </section>
      </div>

      <EnrollmentDialog open={enrollOpen} onClose={() => setEnrollOpen(false)} onComplete={onEnrollmentComplete} />
      <PetDetailDrawer petId={detailPetId} pet={detailPet} loading={detailLoading} error={detailError} onClose={() => { setDetailPetId(null); setDetailPet(null); }} onReload={() => detailPetId ? loadPetDetails(detailPetId) : Promise.resolve()} onChanged={onGalleryChanged} />
      {toast ? <div className="toast" role="status"><span>✓</span>{toast}<button type="button" onClick={() => setToast(null)} aria-label="关闭提示">×</button></div> : null}
    </main>
  );
}
