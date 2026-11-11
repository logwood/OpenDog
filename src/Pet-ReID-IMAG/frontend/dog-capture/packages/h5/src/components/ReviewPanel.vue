<script setup lang="ts">
import { computed } from 'vue';
import type { CaptureSession } from '../lib/session';
import { LEVELS, VIEW_NAMES } from '../lib/texts';

const props = defineProps<{ session: CaptureSession }>();
const emit = defineEmits<{ submit: []; retake: [views: string[]] }>();
const views = computed(() => props.session.config.value!.policy.views.map((v) => {
  const shots = props.session.shots.value.filter((s) => s.view === v.name);
  return { name: v.name, label: VIEW_NAMES[v.name] ?? v.name, need: v.shots || 1, required: v.required, shots };
}));
const missing = computed(() => views.value.filter((v) => v.required && !v.shots.length).map((v) => v.label));
</script>

<template>
  <section class="panel review">
    <p class="eyebrow">确认照片</p>
    <h1>{{ missing.length ? '还缺少一些角度' : '照片已选好' }}</h1>
    <p v-if="missing.length" class="warn">缺少：{{ missing.join('、') }}。可以补拍，或先提交其余照片。</p>
    <div class="grid">
      <article v-for="v in views" :key="v.name" class="card">
        <header><strong>{{ v.label }}</strong><span>{{ v.shots.length }}/{{ v.need }}{{ v.required ? '' : ' · 可选' }}</span></header>
        <div class="thumbs">
          <figure v-for="s in v.shots" :key="s.id">
            <img :src="s.thumbUrl" :alt="`${v.label}照片 ${s.rank}`" />
            <figcaption :class="s.level">{{ LEVELS[s.level] ?? s.level }} · {{ s.score.toFixed(2) }}</figcaption>
          </figure>
          <div v-if="!v.shots.length" class="empty">未采到</div>
        </div>
        <button class="ghost small" @click="emit('retake', [v.name])">重拍{{ v.label }}</button>
      </article>
    </div>
    <p class="muted">缩略图是对齐后的狗脸；上传的是原始高清画面，供识别模型使用。</p>
    <div class="actions">
      <button class="primary" :disabled="!props.session.shots.value.length" @click="emit('submit')">提交</button>
      <button class="ghost" @click="emit('retake', views.map((v) => v.name))">全部重拍</button>
    </div>
  </section>
</template>
