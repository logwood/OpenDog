import type { Metadata, Viewport } from 'next';
import './globals.css';
import './workspace.css';

const siteOrigin = process.env.NEXT_PUBLIC_SITE_ORIGIN ?? 'http://localhost:3000';

export const metadata: Metadata = {
  metadataBase: new URL(siteOrigin),
  title: 'Pawprint ID · 本地识别',
  description: '本地宠物图片录入与身份比对。',
  applicationName: 'Pawprint ID',
  manifest: '/manifest.webmanifest',
  formatDetection: { telephone: false },
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Pawprint ID',
  },
  icons: {
    icon: [
      { url: '/favicon.svg', type: 'image/svg+xml' },
      { url: '/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
      { url: '/icons/icon-512.png', sizes: '512x512', type: 'image/png' },
    ],
    apple: [{ url: '/icons/apple-touch-icon.png', sizes: '180x180', type: 'image/png' }],
  },
  openGraph: {
    title: 'Pawprint ID · 本地识别',
    description: '本地宠物图片录入与身份比对。',
    type: 'website',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'Pawprint ID' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Pawprint ID · 本地识别',
    description: '本地宠物图片录入与身份比对。',
    images: ['/og.png'],
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f4f5f0' },
    { media: '(prefers-color-scheme: dark)', color: '#17201d' },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
