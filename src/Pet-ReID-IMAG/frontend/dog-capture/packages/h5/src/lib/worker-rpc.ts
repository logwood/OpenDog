/** Correlated, bounded worker requests: a crashed worker cannot leave the UI waiting forever. */
export class WorkerRPC {
  private next = 0;
  private closed = false;
  private pending = new Map<number, { resolve(value: unknown): void; reject(error: Error): void;
    timer: ReturnType<typeof setTimeout>; expect?: string }>();

  constructor(private worker: Worker) {
    worker.addEventListener('message', this.message);
    worker.addEventListener('error', this.error);
    worker.addEventListener('messageerror', this.messageError);
  }

  private message = (event: MessageEvent) => {
    const data = event.data;
    const item = this.pending.get(data.requestId);
    if (!item) return;
    if (data.type !== 'error' && item.expect && data.type !== item.expect) return;
    this.pending.delete(data.requestId);
    clearTimeout(item.timer);
    if (data.type === 'error') item.reject(new Error(data.message));
    else item.resolve(data);
  };
  private error = (event: ErrorEvent) => this.close(event.message || 'Inference worker failed');
  private messageError = () => this.close('Inference worker response could not be read');

  call<T>(message: object, transfer: Transferable[] = [], expect?: string, timeoutMs = 30_000): Promise<T> {
    if (this.closed) return Promise.reject(new Error('Inference worker is closed'));
    const requestId = ++this.next;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => this.close('Inference worker timed out; restart capture'), timeoutMs);
      this.pending.set(requestId, { resolve: (value) => resolve(value as T), reject, timer, expect });
      try { this.worker.postMessage({ ...message, requestId }, transfer); }
      catch (error) { this.close(error instanceof Error ? error.message : String(error)); }
    });
  }

  close(reason = 'Capture session closed') {
    if (this.closed) return;
    this.closed = true;
    this.worker.removeEventListener('message', this.message);
    this.worker.removeEventListener('error', this.error);
    this.worker.removeEventListener('messageerror', this.messageError);
    this.worker.terminate();
    for (const item of this.pending.values()) { clearTimeout(item.timer); item.reject(new Error(reason)); }
    this.pending.clear();
  }
}
