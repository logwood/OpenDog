import type { Metadata } from 'next';
import './globals.css';
import './workspace.css';

const siteOrigin = process.env.NEXT_PUBLIC_SITE_ORIGIN ?? 'http://localhost:3000';

export const metadata: Metadata = {
  metadataBase: new URL(siteOrigin),
  title: 'Pawprint ID · 本地识别',
  description: '本地宠物图片录入与身份比对。',
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
