<script setup lang="ts">
import { computed } from 'vue';
import type { SessionConfig } from '@opendog/dog-capture-sdk';
import { VIEW_HINTS, VIEW_NAMES } from '../lib/texts';

const props = defineProps<{ config: SessionConfig | null; loading: boolean; progress: number; message: string }>();
const emit = defineEmits<{ start: []; cancel: [] }>();
const views = computed(() => (props.config?.policy.views ?? []).map((v) => ({
  name: v.name, label: VIEW_NAMES[v.name] ?? v.name, hint: VIEW_HINTS[v.name] ?? '', shots: v.shots || 1, required: v.required,
})));
const purpose = computed(() => (props.config?.purpose === 'identify' ? '识别比对' : '建档录入'));
</script>

<template>
  <section class="panel intro">
    <p class="eyebrow">犬脸采集 · {{ purpose }}</p>
    <h1>请让狗狗看向镜头</h1>
    <ol class="tips">
      <li>光线充足，避免逆光；手机离狗脸 30–50 厘米。</li>
      <li>让狗狗安静坐好，可以点“吸引注意”发出声音。</li>
      <li>按提示依次拍到下面这些角度，系统会自动挑选最清晰的照片。</li>
    </ol>
    <ul class="views">
      <li v-for="v in views" :key="v.name">
        <strong>{{ v.label }}</strong>
        <span>{{ v.hint }}</span>
        <em>{{ v.shots }} 张{{ v.required ? '' : ' · 可选' }}</em>
      </li>
    </ul>
    <p class="privacy">照片只用于本次犬只身份{{ purpose }}；画面在本机实时分析，只上传最终选中的照片。</p>
    <div class="actions">
      <button class="primary" :disabled="loading || !config" @click="emit('start')">
        {{ loading ? `准备中 ${Math.round(progress * 100)}%` : '开始采集' }}
      </button>
      <button class="ghost" @click="emit('cancel')">取消</button>
    </div>
    <p v-if="loading" class="muted">{{ message }}</p>
  </section>
</template>
