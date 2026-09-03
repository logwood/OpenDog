package com.pawprintid.mobile;

import android.content.ContentResolver;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.widget.ImageView;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Decodes bounded image previews away from the main thread. */
public final class ImagePreviewLoader implements Closeable {
    private static final int MAX_PREVIEW_EDGE = 1200;

    private final ContentResolver resolver;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private volatile boolean closed;

    ImagePreviewLoader(Context context) {
        resolver = context.getApplicationContext().getContentResolver();
    }

    void load(Uri uri, ImageView target, Runnable onError) {
        String token = uri.toString();
        target.setTag(token);
        executor.execute(() -> {
            Bitmap bitmap = null;
            try {
                bitmap = decode(uri);
            } catch (IOException | RuntimeException ignored) {
                // The UI receives one localized error instead of provider details.
            }
            Bitmap decoded = bitmap;
            mainHandler.post(() -> {
                if (closed || !token.equals(target.getTag())) {
                    if (decoded != null && !decoded.isRecycled()) {
                        decoded.recycle();
                    }
                    return;
                }
                if (decoded == null) {
                    onError.run();
                } else {
                    target.setImageBitmap(decoded);
                }
            });
        });
    }

    void clear(ImageView target) {
        target.setTag(null);
        target.setImageDrawable(null);
    }

    private Bitmap decode(Uri uri) throws IOException {
        if (Build.VERSION.SDK_INT >= 28) {
            ImageDecoder.Source source = ImageDecoder.createSource(resolver, uri);
            return ImageDecoder.decodeBitmap(source, (decoder, info, sourceInfo) -> {
                int width = info.getSize().getWidth();
                int height = info.getSize().getHeight();
                int largest = Math.max(width, height);
                if (largest > MAX_PREVIEW_EDGE) {
                    float scale = MAX_PREVIEW_EDGE / (float) largest;
                    decoder.setTargetSize(
                            Math.max(1, Math.round(width * scale)),
                            Math.max(1, Math.round(height * scale))
                    );
                }
                decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
            });
        }

        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream input = resolver.openInputStream(uri)) {
            if (input == null) {
                throw new IOException("Unable to open image");
            }
            BitmapFactory.decodeStream(input, null, bounds);
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw new IOException("Unable to read image dimensions");
        }

        int sampleSize = 1;
        while (Math.max(bounds.outWidth / sampleSize, bounds.outHeight / sampleSize)
                > MAX_PREVIEW_EDGE) {
            sampleSize *= 2;
        }
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = sampleSize;
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        try (InputStream input = resolver.openInputStream(uri)) {
            if (input == null) {
                throw new IOException("Unable to open image");
            }
            Bitmap decoded = BitmapFactory.decodeStream(input, null, options);
            if (decoded == null) {
                throw new IOException("Unable to decode image");
            }
            return decoded;
        }
    }

    @Override
    public void close() {
        closed = true;
        executor.shutdownNow();
    }
}
