<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { openCamera, type CameraHandle } from '../lib/camera';
import { playAttention } from '../lib/attention';
import type { CaptureSession } from '../lib/session';
import { VIEW_NAMES } from '../lib/texts';

const props = defineProps<{ session: CaptureSession }>();
const emit = defineEmits<{ finish: []; cancel: []; error: [message: string] }>();
const video = ref<HTMLVideoElement | null>(null);
const camera = ref<CameraHandle | null>(null);
const torchOn = ref(false);
const now = ref(Date.now());
let timer = 0;

const state = props.session.state;
const policy = computed(() => props.session.config.value!.policy);
const guidance = computed(() => state.live?.guidance ?? { code: 'no_face', message: '请把镜头对准狗狗的脸', view: null });
const primary = computed(() => state.live?.faces.find((f) => f.trackId === state.live?.primaryTrack) ?? null);
// Guide colour: green = in a target view and stable; amber = a face with a problem; grey = no usable face.
const tone = computed(() => {
  if (!primary.value) return 'idle';
  if (primary.value.stable && primary.value.quality.passed) return 'good';
  return primary.value.quality.passed ? 'near' : 'warn';
});
const chips = computed(() => policy.value.views.map((v) => {
  const need = v.shots || 1, have = Math.min(state.counts[v.name] ?? 0, need);
  return { name: v.name, label: VIEW_NAMES[v.name] ?? v.name, need, have, required: v.required,
    next: guidance.value.view === v.name, done: have >= need };
}));
const remaining = computed(() => Math.max(0, Math.ceil(policy.value.max_duration_s - (now.value - state.startedAt) / 1000)));
const arrow = computed(() => (guidance.value.code === 'capture_view'
  ? { left: '←', right: '→', front: '●' }[guidance.value.view ?? ''] ?? '' : ''));

onMounted(async () => {
  try {
    camera.value = await openCamera(video.value!);
    props.session.start(video.value!);
    timer = window.setInterval(() => {
      now.value = Date.now();
      if (state.phase === 'error') { emit('error', state.error); return; }
      if (state.complete || remaining.value <= 0) emit('finish');
    }, 250);
  } catch (error) {
    emit('error', error instanceof Error ? error.message : String(error));
  }
});
onBeforeUnmount(() => { window.clearInterval(timer); props.session.pause(); camera.value?.stop(); });

async function toggleTorch() {
  if (camera.value && (await camera.value.setTorch(!torchOn.value))) torchOn.value = !torchOn.value;
}
</script>

<template>
  <section class="stage">
    <video ref="video" class="preview" playsinline muted autoplay></video>
    <div class="guide" :class="tone"><span class="corner tl"></span><span class="corner tr"></span>
      <span class="corner bl"></span><span class="corner br"></span><span v-if="arrow" class="arrow">{{ arrow }}</span></div>
    <header class="topbar">
      <button class="icon" aria-label="取消采集" @click="emit('cancel')">✕</button>
      <span class="timer" :class="{ low: remaining <= 10 }">{{ remaining }} 秒</span>
      <span class="fps" :title="JSON.stringify(state.providers)">{{ state.fps.toFixed(0) }} fps</span>
    </header>
    <div class="prompt" :class="tone" aria-live="polite">{{ guidance.message }}</div>
    <footer class="bottombar">
      <ul class="chips">
        <li v-for="c in chips" :key="c.name" :class="{ done: c.done, next: c.next, optional: !c.required }">
          <span>{{ c.done ? '✓' : '' }} {{ c.label }}</span><small>{{ c.have }}/{{ c.need }}</small>
        </li>
      </ul>
      <div class="controls">
        <button class="ghost" @click="playAttention()">吸引注意</button>
        <button v-if="camera?.torch" class="ghost" @click="toggleTorch()">{{ torchOn ? '关闭补光' : '补光' }}</button>
        <button class="primary" @click="emit('finish')">完成</button>
      </div>
    </footer>
  </section>
</template>
