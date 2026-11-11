import { describe, expect, it, vi } from 'vitest';
import { WorkerRPC } from '../../h5/src/lib/worker-rpc';

class FakeWorker extends EventTarget {
  messages: { requestId: number }[] = [];
  postMessage(message: { requestId: number }) { this.messages.push(message); }
  terminate = vi.fn();
  reply(id: number, type = 'result') { this.dispatchEvent(new MessageEvent('message', { data: { requestId: id, type } })); }
}

describe('capture worker failures', () => {
  it('correlates replies and cancels outstanding work when the page is disposed', async () => {
    const worker = new FakeWorker();
    const rpc = new WorkerRPC(worker as unknown as Worker);
    const a = rpc.call<{ requestId: number }>({ type: 'frame' }, [], 'result');
    const b = rpc.call({ type: 'finish' }, [], 'shots');
    const rejected = expect(b).rejects.toThrow('closed');
    worker.reply(worker.messages[0].requestId);
    expect((await a).requestId).toBe(1);
    rpc.close();
    await rejected;
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it('terminates a stalled worker and rejects the pending request', async () => {
    vi.useFakeTimers();
    try {
      const worker = new FakeWorker();
      const rpc = new WorkerRPC(worker as unknown as Worker);
      const request = rpc.call({ type: 'frame' }, [], 'result', 100);
      const rejected = expect(request).rejects.toThrow('timed out');
      await vi.advanceTimersByTimeAsync(101);
      await rejected;
      expect(worker.terminate).toHaveBeenCalledOnce();
    } finally { vi.useRealTimers(); }
  });
});
