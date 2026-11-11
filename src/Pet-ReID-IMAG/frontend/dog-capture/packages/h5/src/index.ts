/** Library entry: embed the guided capture flow into another Vue 3 app. */
import './styles.css';

export { default as DogCapture } from './components/DogCapture.vue';
export { createCaptureSession, type CaptureSession, type CapturedShot, type LiveResult } from './lib/session';
export { openCamera, CameraError } from './lib/camera';
