/** Rear camera access with operator-readable errors (HTTPS, permission, busy camera). */
export class CameraError extends Error {
  constructor(public code: string, message: string) { super(message); }
}

export interface CameraHandle {
  stream: MediaStream;
  track: MediaStreamTrack;
  settings: MediaTrackSettings;
  torch: boolean;
  setTorch(on: boolean): Promise<boolean>;
  stop(): void;
}

export async function openCamera(video: HTMLVideoElement): Promise<CameraHandle> {
  if (!window.isSecureContext) {
    throw new CameraError('insecure_context', '浏览器只允许在 HTTPS（或本机 localhost）页面中打开相机，请用 https:// 地址访问');
  }
  if (!navigator.mediaDevices?.getUserMedia) throw new CameraError('unsupported', '当前浏览器不支持网页相机，请换用最新版浏览器');
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      // High resolution matters: the original frames are the recognition model's input.
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
    });
  } catch (error) {
    const name = (error as DOMException).name;
    if (name === 'NotAllowedError') throw new CameraError('permission_denied', '没有相机权限，请在浏览器设置中允许本页面使用相机');
    if (name === 'NotFoundError' || name === 'OverconstrainedError') throw new CameraError('no_camera', '没有找到可用的后置相机');
    if (name === 'NotReadableError') throw new CameraError('camera_busy', '相机被其他应用占用，请关闭后重试');
    throw new CameraError('camera_failed', `打开相机失败：${String(error)}`);
  }
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();
  const track = stream.getVideoTracks()[0];
  const capabilities = (track.getCapabilities?.() ?? {}) as MediaTrackCapabilities & { torch?: boolean };
  return {
    stream, track, settings: track.getSettings(), torch: !!capabilities.torch,
    async setTorch(on: boolean) {
      try {
        await track.applyConstraints({ advanced: [{ torch: on } as MediaTrackConstraintSet] });
        return true;
      } catch { return false; }
    },
    stop() { stream.getTracks().forEach((t) => t.stop()); video.srcObject = null; },
  };
}
