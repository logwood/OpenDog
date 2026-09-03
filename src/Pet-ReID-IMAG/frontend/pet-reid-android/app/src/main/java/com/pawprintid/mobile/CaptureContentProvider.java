package com.pawprintid.mobile;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

/**
 * A deliberately small, non-exported provider for full-resolution camera output.
 * It avoids a runtime support-library dependency while retaining content:// URI
 * permission semantics required by modern Android camera applications.
 */
public final class CaptureContentProvider extends ContentProvider {
    private static final String PATH_SEGMENT = "capture";
    private static final String CACHE_DIRECTORY = "camera";

    @Override
    public boolean onCreate() {
        File directory = captureDirectory();
        return directory.exists() || directory.mkdirs();
    }

    public static Uri uriForFile(android.content.Context context, File file) {
        return new Uri.Builder()
                .scheme("content")
                .authority(context.getPackageName() + ".capture")
                .appendPath(PATH_SEGMENT)
                .appendPath(file.getName())
                .build();
    }

    @Override
    public String getType(Uri uri) {
        requireFile(uri);
        return "image/jpeg";
    }

    @Override
    public Cursor query(
            Uri uri,
            String[] projection,
            String selection,
            String[] selectionArgs,
            String sortOrder
    ) {
        File file = requireFile(uri);
        String[] columns = projection == null
                ? new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE}
                : projection;
        MatrixCursor cursor = new MatrixCursor(columns, 1);
        Object[] values = new Object[columns.length];
        for (int index = 0; index < columns.length; index++) {
            if (OpenableColumns.DISPLAY_NAME.equals(columns[index])) {
                values[index] = file.getName();
            } else if (OpenableColumns.SIZE.equals(columns[index])) {
                values[index] = file.length();
            } else {
                values[index] = null;
            }
        }
        cursor.addRow(values);
        return cursor;
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        File file = requireFile(uri);
        if (mode != null && mode.contains("w")) {
            File parent = file.getParentFile();
            if (parent == null || (!parent.exists() && !parent.mkdirs())) {
                throw new FileNotFoundException("Unable to create camera cache directory");
            }
            return ParcelFileDescriptor.open(
                    file,
                    ParcelFileDescriptor.MODE_CREATE
                            | ParcelFileDescriptor.MODE_READ_WRITE
                            | ParcelFileDescriptor.MODE_TRUNCATE
            );
        }
        return ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY);
    }

    @Override
    public int delete(Uri uri, String selection, String[] selectionArgs) {
        return requireFile(uri).delete() ? 1 : 0;
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) {
        throw new UnsupportedOperationException("Insert is not supported");
    }

    @Override
    public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException("Update is not supported");
    }

    private File captureDirectory() {
        if (getContext() == null) {
            throw new IllegalStateException("Provider is not attached");
        }
        return new File(getContext().getCacheDir(), CACHE_DIRECTORY);
    }

    private File requireFile(Uri uri) {
        if (getContext() == null
                || !"content".equals(uri.getScheme())
                || !(getContext().getPackageName() + ".capture").equals(uri.getAuthority())) {
            throw new IllegalArgumentException("Unexpected capture URI");
        }

        List<String> segments = uri.getPathSegments();
        if (segments.size() != 2 || !PATH_SEGMENT.equals(segments.get(0))) {
            throw new IllegalArgumentException("Unexpected capture path");
        }

        String name = segments.get(1);
        if (name.isEmpty() || name.contains("/") || name.contains("\\")) {
            throw new IllegalArgumentException("Invalid capture file name");
        }

        try {
            File directory = captureDirectory().getCanonicalFile();
            File file = new File(directory, name).getCanonicalFile();
            String prefix = directory.getPath() + File.separator;
            if (!file.getPath().startsWith(prefix)) {
                throw new IllegalArgumentException("Capture path leaves cache directory");
            }
            return file;
        } catch (IOException error) {
            throw new IllegalArgumentException("Unable to resolve capture path", error);
        }
    }
}
