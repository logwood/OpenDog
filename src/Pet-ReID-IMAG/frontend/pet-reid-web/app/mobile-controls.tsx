"use client";

import { type FormEvent, useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import {
  DEFAULT_API_BASE_URL,
  getApiBaseUrl,
  normalizeApiBaseUrl,
  setApiBaseUrl,
} from "../lib/pet-reid-api";

type BeforeInstallPromptEvent = Event & {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed"; platform: string }>;
};

function validateApiBaseUrl(value: string): string {
  const normalized = normalizeApiBaseUrl(value);
  if (!normalized) return "";

  let parsed: URL;
  try {
    parsed = new URL(normalized);
  } catch {
    throw new Error("请输入完整地址，例如 http://192.168.1.20:8080。");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("服务地址只支持 http:// 或 https://。");
  }
  if (parsed.username || parsed.password) {
    throw new Error("服务地址中不能包含用户名或密码。");
  }
  if (parsed.search || parsed.hash) {
    throw new Error("服务地址中不能包含查询参数或片段。");
  }
  if (typeof window !== "undefined" && window.location.protocol === "https:" && parsed.protocol === "http:") {
    throw new Error("HTTPS 页面不能直接访问 HTTP 服务，请使用同源代理或为 API 配置 HTTPS。");
  }
  return normalized;
}

function endpointLabel(apiBaseUrl: string): string {
  if (apiBaseUrl) return apiBaseUrl;
  if (typeof window === "undefined") return "当前网页 · 同源 /v1";
  return window.location.origin + " · 同源 /v1";
}

export default function MobileControls({
  onEndpointChanged,
}: {
  onEndpointChanged: () => Promise<void> | void;
}) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [apiBaseUrl, setApiBaseUrlState] = useState(DEFAULT_API_BASE_URL);
  const [draft, setDraft] = useState(DEFAULT_API_BASE_URL);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [installPrompt, setInstallPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [standalone, setStandalone] = useState(false);
  const [secureContext, setSecureContext] = useState(false);

  useEffect(() => {
    const displayMode = window.matchMedia("(display-mode: standalone)");
    const refreshDisplayMode = () => setStandalone(displayMode.matches);
    const initialize = window.setTimeout(() => {
      const configured = getApiBaseUrl();
      setApiBaseUrlState(configured);
      setDraft(configured);
      setSecureContext(window.isSecureContext);
      refreshDisplayMode();
    }, 0);
    displayMode.addEventListener?.("change", refreshDisplayMode);

    const beforeInstall = (event: Event) => {
      event.preventDefault();
      setInstallPrompt(event as BeforeInstallPromptEvent);
    };
    const installed = () => {
      setStandalone(true);
      setInstallPrompt(null);
    };
    window.addEventListener("beforeinstallprompt", beforeInstall);
    window.addEventListener("appinstalled", installed);

    if ("serviceWorker" in navigator && window.isSecureContext) {
      void navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => undefined);
    }

    return () => {
      window.clearTimeout(initialize);
      displayMode.removeEventListener?.("change", refreshDisplayMode);
      window.removeEventListener("beforeinstallprompt", beforeInstall);
      window.removeEventListener("appinstalled", installed);
    };
  }, []);

  const currentEndpoint = useMemo(() => endpointLabel(apiBaseUrl), [apiBaseUrl]);

  const openDialog = () => {
    setDraft(apiBaseUrl);
    setError(null);
    setDialogOpen(true);
  };

  const install = async () => {
    if (!installPrompt) {
      setDialogOpen(true);
      return;
    }
    await installPrompt.prompt();
    const choice = await installPrompt.userChoice;
    if (choice.outcome === "accepted") setInstallPrompt(null);
  };

  const applyEndpoint = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const normalized = validateApiBaseUrl(draft);
      const configured = setApiBaseUrl(normalized);
      setApiBaseUrlState(configured);
      setDraft(configured);
      await onEndpointChanged();
      setDialogOpen(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "服务地址没有保存，请重试。");
    } finally {
      setSaving(false);
    }
  };

  const restoreDefault = async () => {
    setSaving(true);
    setError(null);
    try {
      const configured = setApiBaseUrl(null);
      setApiBaseUrlState(configured);
      setDraft(configured);
      await onEndpointChanged();
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      {!standalone && installPrompt ? (
        <button className="install-app-button" type="button" onClick={() => void install()}>
          安装应用
        </button>
      ) : null}
      <button
        className="mobile-settings-button"
        type="button"
        onClick={openDialog}
        title={currentEndpoint}
        aria-label="打开手机连接与安装设置"
      >
        <span aria-hidden="true">⌁</span><b>手机连接</b>
      </button>

      {dialogOpen && typeof document !== "undefined" ? createPortal((
        <div
          className="dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => event.target === event.currentTarget && setDialogOpen(false)}
        >
          <section className="dialog-card mobile-setup-dialog" role="dialog" aria-modal="true" aria-labelledby="mobile-setup-title">
            <div className="dialog-heading">
              <div>
                <p className="eyebrow">Web + Android</p>
                <h2 id="mobile-setup-title">手机连接</h2>
              </div>
              <button className="icon-button" type="button" onClick={() => setDialogOpen(false)} aria-label="关闭手机连接设置">×</button>
            </div>

            <div className="mobile-install-card">
              <span className="mobile-install-icon" aria-hidden="true">P</span>
              <div>
                <strong>{standalone ? "已作为应用运行" : "安装 Pawprint ID"}</strong>
                <p>
                  {standalone
                    ? "当前已使用独立窗口，可直接拍照识别。"
                    : installPrompt
                      ? "安装后会出现在安卓桌面，并以独立窗口运行。"
                      : secureContext
                        ? "在 Chrome 菜单中选择“安装应用”或“添加到主屏幕”。"
                        : "当前局域网 HTTP 地址可正常使用；完整应用安装需要 HTTPS，Chrome 菜单仍可添加桌面快捷方式。"}
                </p>
              </div>
              {!standalone && installPrompt ? <button type="button" onClick={() => void install()}>立即安装</button> : null}
            </div>

            <form className="endpoint-form" onSubmit={applyEndpoint}>
              <div className="endpoint-current">
                <span>当前请求目标</span>
                <strong title={currentEndpoint}>{currentEndpoint}</strong>
              </div>
              <label>
                <span>API 根地址</span>
                <input
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  inputMode="url"
                  autoCapitalize="none"
                  autoCorrect="off"
                  spellCheck={false}
                  placeholder="留空即使用当前网页的同源 /v1 代理"
                />
              </label>
              <p className="endpoint-help">
                手机访问电脑前端时推荐留空；如直连网关，请填写电脑局域网地址，例如 http://192.168.1.20:8080，不要附加 /v1。
              </p>
              <div className="endpoint-presets">
                <button type="button" onClick={() => setDraft("")}>使用同源代理</button>
                <button type="button" onClick={() => void restoreDefault()} disabled={saving}>恢复构建默认</button>
              </div>
              {error ? <p className="form-error" role="alert">{error}</p> : null}
              <div className="dialog-actions">
                <button className="secondary-button" type="button" onClick={() => setDialogOpen(false)} disabled={saving}>取消</button>
                <button className="primary-button" type="submit" disabled={saving}>{saving ? "连接中…" : "保存并重连"}</button>
              </div>
            </form>
          </section>
        </div>
      ), document.body) : null}
    </>
  );
}
