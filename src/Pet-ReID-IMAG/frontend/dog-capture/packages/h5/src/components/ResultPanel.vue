<script setup lang="ts">
import { computed, ref } from 'vue';
import type { CaptureSession } from '../lib/session';
import { reasonText, VIEW_NAMES } from '../lib/texts';

const props = defineProps<{ session: CaptureSession }>();
const emit = defineEmits<{ retake: [views: string[]]; close: [] }>();
const result = computed(() => props.session.result.value!);
const accepted = computed(() => result.value.status === 'accepted');
const rejected = computed(() => result.value.shots.filter((s) => !s.accepted));
const missingText = computed(() => result.value.missing_views.map((name: string) => VIEW_NAMES[name] ?? name).join('、'));
const committed = ref<unknown>(null);
const committing = ref(false);
const error = ref('');

async function commit() {
  committing.value = true;
  try { committed.value = await props.session.commit(); } catch (e) { error.value = e instanceof Error ? e.message : String(e); }
  committing.value = false;
}
</script>

<template>
  <section class="panel result">
    <div class="badge" :class="accepted ? 'ok' : 'retry'">{{ accepted ? '✓' : '!' }}</div>
    <h1>{{ accepted ? '采集成功' : '有照片需要补拍' }}</h1>
    <p v-if="accepted" class="muted">
      服务器已复核 {{ result.shots.filter((s) => s.accepted).length }} 张照片。
      {{ result.capture_token ? '可以直接完成。' : '请回到业务页面继续，后续由系统完成录入或比对。' }}
    </p>
    <ul v-if="rejected.length" class="reasons">
      <li v-for="s in rejected" :key="s.id"><strong>{{ VIEW_NAMES[s.view] ?? s.view }}</strong>：{{ reasonText(s.reasons) }}</li>
    </ul>
    <p v-if="missingText" class="warn">缺少：{{ missingText }}</p>
    <div class="actions">
      <button v-if="!accepted" class="primary" @click="emit('retake', result.retake_views)">补拍这些角度</button>
      <button v-if="accepted && result.capture_token && !committed" class="primary" :disabled="committing" @click="commit">
        {{ committing ? '提交中…' : props.session.config.value?.purpose === 'identify' ? '开始比对' : '完成建档' }}
      </button>
      <button class="ghost" @click="emit('close')">关闭</button>
    </div>
    <pre v-if="committed" class="json">{{ JSON.stringify(committed, null, 2) }}</pre>
    <p v-if="error" class="warn">{{ error }}</p>
  </section>
</template>
