/** A short squeaky-toy chirp (WebAudio, no asset) to make the dog look at the phone. */
let context: AudioContext | null = null;

export function playAttention() {
  context = context ?? new AudioContext();
  const now = context.currentTime;
  for (let i = 0; i < 3; i++) {
    const osc = context.createOscillator();
    const gain = context.createGain();
    const start = now + i * 0.16;
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(1400, start);
    osc.frequency.exponentialRampToValueAtTime(2600, start + 0.09);
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.35, start + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.12);
    osc.connect(gain).connect(context.destination);
    osc.start(start);
    osc.stop(start + 0.13);
  }
}
