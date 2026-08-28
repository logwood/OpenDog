"use client";

/* eslint-disable @next/next/no-img-element */
import {
  type ChangeEvent,
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type BatchJob,
  type HistoryItem,
  type HistoryListResponse,
  type ReviewStatus,
  errorMessage,
  petReIdApi,
} from "../lib/pet-reid-api";

function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function metric(value: number | null | undefined, digits = 4): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

function percent(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? (value * 100).toFixed(1) + "%"
    : "—";
}

const REVIEW_LABEL: Record<ReviewStatus, string> = {
  unreviewed: "未复核",
  correct: "正确",
  incorrect: "错误",
  uncertain: "不确定",
};

const HARD_LABEL: Record<string, string> = {
  rejected: "被拒识",
  low_margin: "差值较小",
  branch_conflict: "鼻脸冲突",
  single_branch: "单分支",
  low_quality: "质量较低",
  incorrect_top1: "Top-1 错误",
  processing_error: "处理失败",
};

function historyIdentity(item: HistoryItem): string {
  if (item.status === "failed") return "处理失败";
  if (!item.accepted) return "未确认身份";
  return item.predicted_display_name || item.predicted_pet_id || "未确认身份";
}

function HistoryRows({ items, onOpen }: {
  items: HistoryItem[];
  onOpen: (historyId: string) => void;
}) {
  if (!items.length) return <p className="section-empty">没有符合条件的记录。</p>;
  return (
    <div className="history-list">
      {items.map((item) => (
        <article className="history-row" key={item.history_id}>
          <div className="history-main">
            <div className="history-title-line">
              <strong>{historyIdentity(item)}</strong>
              <span className={"review-chip " + item.review_status}>{REVIEW_LABEL[item.review_status]}</span>
            </div>
            <small>{item.filename} · {item.source === "batch" ? "批量" : "单张"} · {formatDate(item.created_at)}</small>
            {item.hard_case_reasons.length ? (
              <div className="reason-tags">
                {item.hard_case_reasons.map((reason) => <span key={reason}>{HARD_LABEL[reason] ?? reason}</span>)}
              </div>
            ) : null}
          </div>
          <div className="history-metrics">
            <span>分数 <strong>{metric(item.top1_score)}</strong></span>
            <span>差值 <strong>{metric(item.margin)}</strong></span>
            <span>耗时 <strong>{item.latency_ms == null ? "—" : Math.round(item.latency_ms) + " ms"}</strong></span>
          </div>
          <button className="row-button" type="button" onClick={() => onOpen(item.history_id)}>查看</button>
        </article>
      ))}
    </div>
  );
}

function HistoryDialog({ item, onClose, onChanged }: {
  item: HistoryItem;
  onClose: () => void;
  onChanged: (item: HistoryItem | null) => void;
}) {
  const [note, setNote] = useState(item.review_note ?? "");
  const [busy, setBusy] = useState(false);
  const result = item.result;

  const review = async (status: ReviewStatus) => {
    setBusy(true);
    try {
      onChanged(await petReIdApi.reviewHistory(item.history_id, status, note));
    } catch (error) {
      window.alert(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!window.confirm("确认删除这条比对历史吗？")) return;
    setBusy(true);
    try {
      await petReIdApi.deleteHistory(item.history_id);
      onChanged(null);
      onClose();
    } catch (error) {
      window.alert(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="dialog-card history-dialog" role="dialog" aria-modal="true" aria-labelledby="history-detail-title">
        <div className="dialog-heading">
          <div><h2 id="history-detail-title">比对详情</h2><small>{item.history_id.slice(0, 12)} · {formatDate(item.created_at)}</small></div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="关闭历史详情">×</button>
        </div>
        <div className="history-detail-grid">
          <div className="history-image">
            {item.image_available ? <img src={petReIdApi.historyImageUrl(item.history_id)} alt={item.filename} /> : <span>没有保存图片</span>}
          </div>
          <div className="history-detail-summary">
            <h3>{historyIdentity(item)}</h3>
            <dl>
              <div><dt>文件</dt><dd>{item.filename}</dd></div>
              <div><dt>预期身份</dt><dd>{item.expected_pet_id ?? "未标注"}</dd></div>
              <div><dt>Top-1 分数</dt><dd>{metric(item.top1_score)}</dd></div>
              <div><dt>差值</dt><dd>{metric(item.margin)}</dd></div>
              <div><dt>耗时</dt><dd>{item.latency_ms == null ? "—" : Math.round(item.latency_ms) + " ms"}</dd></div>
              <div><dt>模型</dt><dd title={item.model_fingerprint}>{item.model_fingerprint.slice(0, 12)}</dd></div>
              <div><dt>图库快照</dt><dd>{item.gallery_snapshot.pets} 个身份 / {item.gallery_snapshot.reference_images} 张图</dd></div>
            </dl>
          </div>
        </div>
        {item.error ? <p className="form-error">{item.error.message || item.error.code}</p> : null}
        {result?.diagnostics?.branch_top1 ? (
          <div className="branch-diagnostics">
            <strong>分支诊断</strong>
            <span>鼻子：{result.diagnostics.branch_top1.nose?.display_name || result.diagnostics.branch_top1.nose?.pet_id || "不可用"} · {metric(result.diagnostics.branch_top1.nose?.score)}</span>
            <span>脸部：{result.diagnostics.branch_top1.face?.display_name || result.diagnostics.branch_top1.face?.pet_id || "不可用"} · {metric(result.diagnostics.branch_top1.face?.score)}</span>
          </div>
        ) : null}
        {result?.candidates?.length ? (
          <div className="history-candidates">
            {result.candidates.map((candidate, index) => (
              <div key={candidate.pet_id}><span>{index + 1}</span><strong>{candidate.display_name || candidate.pet_id}</strong><small>{metric(candidate.score)}</small></div>
            ))}
          </div>
        ) : null}
        <div className="review-box">
          <label><span>复核备注</span><textarea value={note} maxLength={1000} onChange={(event) => setNote(event.target.value)} placeholder="可选" /></label>
          <div className="review-actions">
            <button type="button" disabled={busy} onClick={() => void review("correct")}>正确</button>
            <button type="button" disabled={busy} onClick={() => void review("incorrect")}>错误</button>
            <button type="button" disabled={busy} onClick={() => void review("uncertain")}>不确定</button>
            <button className="secondary-button" type="button" disabled={busy} onClick={() => void review("unreviewed")}>清除标记</button>
          </div>
        </div>
        <div className="dialog-actions"><button className="danger-button" type="button" disabled={busy} onClick={() => void remove()}>删除记录</button></div>
      </section>
    </div>
  );
}

export function HistorySection({ refreshToken = 0 }: { refreshToken?: number }) {
  const [data, setData] = useState<HistoryListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState("");
  const [reviewStatus, setReviewStatus] = useState("");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<HistoryItem | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await petReIdApi.listHistory({ page, pageSize: 20, source, reviewStatus }));
      setError(null);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }, [page, reviewStatus, source]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load, refreshToken]);

  const open = async (historyId: string) => {
    try { setSelected(await petReIdApi.getHistory(historyId)); }
    catch (requestError) { window.alert(errorMessage(requestError)); }
  };

  const changed = (item: HistoryItem | null) => {
    setSelected(item);
    void load();
  };

  const pages = Math.max(1, Math.ceil((data?.total ?? 0) / 20));
  return (
    <section className="tool-section" id="history">
      <div className="section-title">
        <div><h2>比对历史</h2><p>{data?.total ?? 0} 条记录</p></div>
        <div className="filter-row">
          <select value={source} onChange={(event) => { setSource(event.target.value); setPage(1); }} aria-label="记录来源">
            <option value="">全部来源</option><option value="single">单张比对</option><option value="batch">批量测试</option>
          </select>
          <select value={reviewStatus} onChange={(event) => { setReviewStatus(event.target.value); setPage(1); }} aria-label="复核状态">
            <option value="">全部复核状态</option><option value="unreviewed">未复核</option><option value="correct">正确</option><option value="incorrect">错误</option><option value="uncertain">不确定</option>
          </select>
          <button className="refresh-button" type="button" onClick={() => void load()} disabled={loading} aria-label="刷新历史">↻</button>
        </div>
      </div>
      {error ? <p className="form-error">{error}</p> : null}
      {loading && !data ? <p className="section-empty">读取历史…</p> : <HistoryRows items={data?.items ?? []} onOpen={(id) => void open(id)} />}
      {pages > 1 ? <div className="pagination"><button type="button" disabled={page <= 1} onClick={() => setPage(page - 1)}>上一页</button><span>{page} / {pages}</span><button type="button" disabled={page >= pages} onClick={() => setPage(page + 1)}>下一页</button></div> : null}
      {selected ? <HistoryDialog item={selected} onClose={() => setSelected(null)} onChanged={changed} /> : null}
    </section>
  );
}

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function batchLabel(file: File): string | null {
  const relative = (file as File & { webkitRelativePath?: string }).webkitRelativePath ?? "";
  const parts = relative.split("/").filter(Boolean);
  return parts.length > 1 ? parts[parts.length - 2] : null;
}

function BatchCard({ job, adminKey, onChanged }: {
  job: BatchJob;
  adminKey: string;
  onChanged: () => Promise<void>;
}) {
  const progress = job.total ? Math.min(100, (job.completed / job.total) * 100) : 0;
  const active = job.status === "queued" || job.status === "running";
  const exportCsv = async () => {
    try { saveBlob(await petReIdApi.downloadBatchCsv(adminKey, job.batch_id), job.name + ".csv"); }
    catch (error) { window.alert(errorMessage(error)); }
  };
  const cancel = async () => {
    if (!window.confirm("确认取消这个批量任务吗？已完成的记录会保留。")) return;
    try { await petReIdApi.cancelBatch(adminKey, job.batch_id); await onChanged(); }
    catch (error) { window.alert(errorMessage(error)); }
  };
  return (
    <article className="batch-card">
      <div className="batch-title"><div><strong>{job.name}</strong><small>{formatDate(job.created_at)} · {job.status}</small></div><span>{job.completed} / {job.total}</span></div>
      <div className="progress-track"><i style={{ width: progress + "%" }} /></div>
      <div className="batch-metrics">
        <span>Top-1 <strong>{percent(job.metrics.top1_accuracy)}</strong></span>
        <span>拒识 <strong>{job.metrics.rejected ?? "—"}</strong></span>
        <span>难例 <strong>{job.metrics.hard_cases ?? "—"}</strong></span>
        <span>平均耗时 <strong>{job.metrics.average_latency_ms == null ? "—" : Math.round(job.metrics.average_latency_ms) + " ms"}</strong></span>
      </div>
      {job.error_message ? <p className="form-error">{job.error_message}</p> : null}
      <div className="batch-actions">
        <button type="button" disabled={active} onClick={() => void exportCsv()}>导出 CSV</button>
        {active ? <button className="danger-button" type="button" onClick={() => void cancel()}>取消</button> : null}
      </div>
    </article>
  );
}

export function AdminSection({ onGalleryChanged }: { onGalleryChanged: (message: string) => Promise<void> }) {
  const [keyInput, setKeyInput] = useState("");
  const [adminKey, setAdminKey] = useState("");
  const [authorized, setAuthorized] = useState(false);
  const [unlocking, setUnlocking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [batchName, setBatchName] = useState("批量测试");
  const [submitting, setSubmitting] = useState(false);
  const [batches, setBatches] = useState<BatchJob[]>([]);
  const [hardCases, setHardCases] = useState<HistoryItem[]>([]);
  const [selectedHardCase, setSelectedHardCase] = useState<HistoryItem | null>(null);
  const [restoreFile, setRestoreFile] = useState<File | null>(null);
  const directoryRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    directoryRef.current?.setAttribute("webkitdirectory", "");
    directoryRef.current?.setAttribute("directory", "");
  }, []);

  const loadAdmin = useCallback(async (key = adminKey) => {
    if (!key) return;
    const [batchData, hardData] = await Promise.all([
      petReIdApi.listBatches(key),
      petReIdApi.listHardCases(key),
    ]);
    setBatches(batchData.items);
    setHardCases(hardData.items);
  }, [adminKey]);

  const unlock = useCallback(async (value: string) => {
    const key = value.trim();
    if (!key) return;
    setUnlocking(true); setError(null);
    try {
      await petReIdApi.adminAccess(key);
      setAdminKey(key); setKeyInput(""); setAuthorized(true);
      window.sessionStorage.setItem("pet-reid-admin-key", key);
      await loadAdmin(key);
    } catch (requestError) {
      setAuthorized(false);
      window.sessionStorage.removeItem("pet-reid-admin-key");
      setError(errorMessage(requestError));
    } finally {
      setUnlocking(false);
    }
  }, [loadAdmin]);

  useEffect(() => {
    const saved = window.sessionStorage.getItem("pet-reid-admin-key");
    const timer = saved ? window.setTimeout(() => void unlock(saved), 0) : null;
    return () => { if (timer !== null) window.clearTimeout(timer); };
  }, [unlock]);

  const activeJobs = batches.some((job) => job.status === "queued" || job.status === "running");
  useEffect(() => {
    if (!authorized || !activeJobs) return;
    const timer = window.setInterval(() => void loadAdmin().catch(() => undefined), 2000);
    return () => window.clearInterval(timer);
  }, [activeJobs, authorized, loadAdmin]);

  const selectedLabels = useMemo(() => files.map(batchLabel), [files]);
  const labelledCount = selectedLabels.filter(Boolean).length;
  const chooseBatchFiles = (incoming: File[]) => {
    const valid = incoming.filter((file) => file.size > 0 && file.size <= 15 * 1024 * 1024);
    if (valid.length > 1000) { setError("一个批次最多 1000 张图片。"); return; }
    const total = valid.reduce((sum, file) => sum + file.size, 0);
    if (total > 110 * 1024 * 1024) { setError("单个批次的图片总量不能超过 110 MB。"); return; }
    setFiles(valid); setError(valid.length === incoming.length ? null : "已跳过空文件或超过 15 MB 的图片。");
  };

  const submitBatch = async (event: FormEvent) => {
    event.preventDefault();
    if (!files.length) { setError("请先选择测试图片或测试集目录。"); return; }
    setSubmitting(true); setError(null);
    try {
      await petReIdApi.createBatch(adminKey, batchName, files, selectedLabels, 5);
      setFiles([]); setBatchName("批量测试");
      await loadAdmin();
    } catch (requestError) { setError(errorMessage(requestError)); }
    finally { setSubmitting(false); }
  };

  const downloadBackup = async () => {
    try { saveBlob(await petReIdApi.downloadGalleryBackup(adminKey), "pet-reid-gallery.zip"); }
    catch (requestError) { setError(errorMessage(requestError)); }
  };

  const restore = async () => {
    if (!restoreFile) return;
    setSubmitting(true); setError(null);
    try {
      const response = await petReIdApi.restoreGallery(adminKey, restoreFile);
      setRestoreFile(null);
      await onGalleryChanged("图库恢复完成：新增 " + response.added_images + " 张，重复 " + response.duplicate_images + " 张。");
    } catch (requestError) { setError(errorMessage(requestError)); }
    finally { setSubmitting(false); }
  };

  const openHardCase = async (historyId: string) => {
    try { setSelectedHardCase(await petReIdApi.getHistory(historyId)); }
    catch (requestError) { setError(errorMessage(requestError)); }
  };

  if (!authorized) {
    return (
      <section className="tool-section admin-locked" id="admin">
        <div className="section-title"><div><h2>管理员工具</h2><p>批量测试、难例与图库备份</p></div></div>
        <form onSubmit={(event) => { event.preventDefault(); void unlock(keyInput); }}>
          <label><span>管理员密钥</span><input type="password" value={keyInput} onChange={(event) => setKeyInput(event.target.value)} placeholder="logs/quick_start/admin-key.txt" autoComplete="off" /></label>
          <button type="submit" disabled={unlocking || !keyInput.trim()}>{unlocking ? "验证中…" : "解锁"}</button>
        </form>
        {error ? <p className="form-error">{error}</p> : null}
      </section>
    );
  }

  return (
    <section className="tool-section admin-tools" id="admin">
      <div className="section-title"><div><h2>管理员工具</h2><p>已验证当前会话</p></div><button className="secondary-button" type="button" onClick={() => { setAuthorized(false); setAdminKey(""); window.sessionStorage.removeItem("pet-reid-admin-key"); }}>锁定</button></div>
      {error ? <p className="form-error">{error}</p> : null}

      <div className="admin-grid">
        <section className="admin-card batch-create">
          <h3>新建批量测试</h3>
          <p>选择普通图片时不计算准确率；选择按身份分目录的测试集时，以一级目录名作为预期身份。</p>
          <form onSubmit={submitBatch}>
            <label><span>任务名称</span><input value={batchName} maxLength={128} onChange={(event) => setBatchName(event.target.value)} /></label>
            <div className="batch-file-buttons">
              <label className="file-button">选择图片<input className="sr-only" type="file" multiple accept="image/jpeg,image/png,image/webp,image/bmp" onChange={(event) => chooseBatchFiles(Array.from(event.target.files ?? []))} /></label>
              <label className="file-button">选择测试集目录<input ref={directoryRef} className="sr-only" type="file" multiple accept="image/jpeg,image/png,image/webp,image/bmp" onChange={(event) => chooseBatchFiles(Array.from(event.target.files ?? []))} /></label>
            </div>
            <p className="file-summary">{files.length ? files.length + " 张 · 已识别标签 " + labelledCount + " 张" : "尚未选择文件"}</p>
            <button className="primary-button" type="submit" disabled={submitting || !files.length}>{submitting ? "正在提交…" : "开始后台任务"}</button>
          </form>
        </section>

        <section className="admin-card backup-card">
          <h3>图库备份</h3>
          <p>备份包含身份、参考图和模型指纹。恢复采用合并模式，不覆盖已有身份。</p>
          <button type="button" onClick={() => void downloadBackup()}>下载备份</button>
          <label className="restore-file"><span>恢复文件</span><input type="file" accept="application/zip,.zip" onChange={(event: ChangeEvent<HTMLInputElement>) => setRestoreFile(event.target.files?.[0] ?? null)} /></label>
          <button type="button" disabled={!restoreFile || submitting} onClick={() => void restore()}>合并恢复</button>
        </section>
      </div>

      <section className="admin-subsection">
        <div className="subsection-heading"><div><h3>批量任务</h3><span>{batches.length}</span></div><button className="refresh-button" type="button" onClick={() => void loadAdmin()} aria-label="刷新管理员数据">↻</button></div>
        <div className="batch-list">{batches.map((job) => <BatchCard key={job.batch_id} job={job} adminKey={adminKey} onChanged={loadAdmin} />)}{!batches.length ? <p className="section-empty">还没有批量任务。</p> : null}</div>
      </section>

      <section className="admin-subsection">
        <div className="subsection-heading"><div><h3>难例</h3><span>{hardCases.length}</span></div></div>
        <HistoryRows items={hardCases} onOpen={(historyId) => void openHardCase(historyId)} />
      </section>
      {selectedHardCase ? <HistoryDialog item={selectedHardCase} onClose={() => setSelectedHardCase(null)} onChanged={(item) => { setSelectedHardCase(item); void loadAdmin(); }} /> : null}
    </section>
  );
}
