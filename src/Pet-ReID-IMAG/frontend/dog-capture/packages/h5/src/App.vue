<script setup lang="ts">
/**
 * Standalone page: /capture/?session=…&nonce=…  (the business backend creates
 * the session and opens this URL, like a face-authentication H5).  Without
 * those parameters a developer launcher creates a session on a key-less server.
 */
import { ref } from 'vue';
import DogCapture from './components/DogCapture.vue';

const params = new URLSearchParams(location.search);
const sessionId = ref(params.get('session') ?? '');
const nonce = ref(params.get('nonce') ?? '');
const purpose = ref<'enroll' | 'identify'>('identify');
const petId = ref('');
const apiKey = ref('');
const error = ref('');

async function launch() {
  error.value = '';
  try {
    const response = await fetch('/v1/capture/sessions', {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(apiKey.value ? { 'x-api-key': apiKey.value } : {}) },
      body: JSON.stringify({ purpose: purpose.value, pet_id: petId.value || undefined, client_commit: true }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.message ?? body.detail?.message ?? `HTTP ${response.status}`);
    history.replaceState(null, '', `?session=${body.session_id}&nonce=${body.nonce}`);
    sessionId.value = body.session_id;
    nonce.value = body.nonce;
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}
</script>

<template>
  <DogCapture v-if="sessionId && nonce" :session-id="sessionId" :nonce="nonce" />
  <section v-else class="panel launcher">
    <p class="eyebrow">开发调试入口</p>
    <h1>新建采集会话</h1>
    <p class="muted">正式接入时由业务后端调用 <code>POST /v1/capture/sessions</code>，再打开
      <code>/capture/?session=…&amp;nonce=…</code>。这里仅用于本机调试。</p>
    <label>用途
      <select v-model="purpose"><option value="identify">识别比对</option><option value="enroll">建档录入</option></select>
    </label>
    <label v-if="purpose === 'enroll'">狗狗 ID <input v-model="petId" placeholder="dog-001" /></label>
    <label>API Key（服务端开启鉴权时）<input v-model="apiKey" type="password" autocomplete="off" /></label>
    <button class="primary" :disabled="purpose === 'enroll' && !petId" @click="launch">开始</button>
    <p v-if="error" class="warn">{{ error }}</p>
  </section>
</template>
