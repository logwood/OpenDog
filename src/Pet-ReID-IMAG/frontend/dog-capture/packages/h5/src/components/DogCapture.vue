<script setup lang="ts">
/**
 * Embeddable guided capture: <DogCapture api-base="" session-id="…" nonce="…" @finished @cancel @error />.
 * Also notifies a parent frame (postMessage) and an Android/iOS WebView bridge
 * (window.DogCaptureBridge.onResult / webkit.messageHandlers.dogCapture), so
 * existing apps can embed the page without sharing code.
 */
import { onBeforeUnmount, onMounted, ref } from 'vue';
import type { VerificationResult } from '@opendog/dog-capture-sdk';
import { createCaptureSession } from '../lib/session';
import CameraStage from './CameraStage.vue';
import IntroPanel from './IntroPanel.vue';
import ResultPanel from './ResultPanel.vue';
import ReviewPanel from './ReviewPanel.vue';

const props = withDefaults(defineProps<{ apiBase?: string; sessionId: string; nonce: string }>(), { apiBase: '' });
const emit = defineEmits<{ finished: [result: VerificationResult]; cancel: []; error: [message: string] }>();
const session = createCaptureSession({ apiBase: props.apiBase, sessionId: props.sessionId, nonce: props.nonce });
const screen = ref<'intro' | 'camera' | 'review' | 'result' | 'error'>('intro');
const error = ref('');

function notify(type: string, payload: object) {
  const message = { source: 'dog-capture', type, session_id: props.sessionId, ...payload };
  window.parent?.postMessage(message, '*');
  const bridge = (window as unknown as { DogCaptureBridge?: { onResult(json: string): void } }).DogCaptureBridge;
  bridge?.onResult(JSON.stringify(message));
  (window as unknown as { webkit?: { messageHandlers?: { dogCapture?: { postMessage(m: object): void } } } })
    .webkit?.messageHandlers?.dogCapture?.postMessage(message);
}

function fail(message: string) {
  error.value = message;
  screen.value = 'error';
  emit('error', message);
  notify('error', { message });
}

onMounted(async () => {
  await session.load();
  if (session.state.phase === 'error') fail(session.state.error);
});
onBeforeUnmount(() => session.dispose());

async function finish() {
  if (screen.value !== 'camera') return;
  screen.value = 'review';
  try { await session.finish(); } catch (error) { fail(String(error)); }
}
async function retake(views: string[]) {
  try { await session.retake(views); screen.value = 'camera'; } catch (error) { fail(String(error)); }
}
async function submit() {
  await session.upload();
  if (session.state.phase === 'error') return fail(session.state.error);
  screen.value = 'result';
  const result = session.result.value!;
  emit('finished', result);
  notify('result', { status: result.status, retake_views: result.retake_views });
}
function cancel() { notify('cancel', {}); emit('cancel'); screen.value = 'intro'; session.pause(); }
</script>

<template>
  <div class="dog-capture">
    <IntroPanel v-if="screen === 'intro'" :config="session.config.value" :loading="session.state.phase === 'loading'"
      :progress="session.state.progress" :message="session.state.message" @start="screen = 'camera'" @cancel="cancel" />
    <CameraStage v-else-if="screen === 'camera'" :session="session" @finish="finish" @cancel="cancel" @error="fail" />
    <ReviewPanel v-else-if="screen === 'review' && session.state.phase === 'review'" :session="session"
      @submit="submit" @retake="retake" />
    <section v-else-if="screen === 'review'" class="panel"><p class="muted">正在整理照片…</p></section>
    <ResultPanel v-else-if="screen === 'result'" :session="session" @retake="retake" @close="cancel" />
    <section v-else class="panel error">
      <h1>无法继续</h1>
      <p class="warn">{{ error }}</p>
      <button class="ghost" @click="screen = 'intro'">返回</button>
    </section>
    <div v-if="session.state.phase === 'uploading'" class="overlay"><p>正在上传并由服务器复核…</p></div>
  </div>
</template>
