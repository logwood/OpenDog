package com.pawprintid.mobile;

import android.content.ContentResolver;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.provider.OpenableColumns;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Native HTTP client for the computer-hosted Pet ReID API.
 *
 * <p>This class deliberately uses Android/Java platform APIs only. No browser,
 * WebView, JavaScript bridge, or web asset is involved.</p>
 */
public final class PetReIdApiClient implements Closeable {
    private static final int CONNECT_TIMEOUT_MS = 8_000;
    private static final int READ_TIMEOUT_MS = 120_000;
    private static final int MAX_JSON_BYTES = 4 * 1024 * 1024;

    public interface JsonCallback {
        void onSuccess(JSONObject response);

        void onError(ApiError error);
    }

    public static final class ApiError extends Exception {
        public final int status;
        public final String code;

        ApiError(String message, int status, String code, Throwable cause) {
            super(message, cause);
            this.status = status;
            this.code = code == null || code.isEmpty() ? "request_failed" : code;
        }
    }

    public static final class FileInfo {
        public final String name;
        public final String contentType;
        public final long size;

        FileInfo(String name, String contentType, long size) {
            this.name = name;
            this.contentType = contentType;
            this.size = size;
        }
    }

    private interface JsonRequest {
        JSONObject run() throws Exception;
    }

    private final ContentResolver resolver;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final ExecutorService executor = Executors.newFixedThreadPool(2);
    private volatile String baseUrl;
    private volatile boolean closed;

    public PetReIdApiClient(Context context, String baseUrl) {
        resolver = context.getApplicationContext().getContentResolver();
        setBaseUrl(baseUrl);
    }

    public void setBaseUrl(String value) {
        String normalized = value == null ? "" : value.trim();
        while (normalized.endsWith("/")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        baseUrl = normalized;
    }

    public FileInfo inspect(Uri uri) {
        String name = "pet-image.jpg";
        long size = -1L;
        try (Cursor cursor = resolver.query(
                uri,
                new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE},
                null,
                null,
                null
        )) {
            if (cursor != null && cursor.moveToFirst()) {
                int nameColumn = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                int sizeColumn = cursor.getColumnIndex(OpenableColumns.SIZE);
                if (nameColumn >= 0 && !cursor.isNull(nameColumn)) {
                    String candidate = cursor.getString(nameColumn);
                    if (candidate != null && !candidate.trim().isEmpty()) {
                        name = candidate.trim();
                    }
                }
                if (sizeColumn >= 0 && !cursor.isNull(sizeColumn)) {
                    size = cursor.getLong(sizeColumn);
                }
            }
        } catch (RuntimeException ignored) {
            // Some document providers expose the stream but not metadata.
        }

        String contentType = resolver.getType(uri);
        if (contentType == null || !contentType.toLowerCase(Locale.US).startsWith("image/")) {
            String lower = name.toLowerCase(Locale.US);
            if (lower.endsWith(".png")) {
                contentType = "image/png";
            } else if (lower.endsWith(".webp")) {
                contentType = "image/webp";
            } else if (lower.endsWith(".bmp")) {
                contentType = "image/bmp";
            } else {
                contentType = "image/jpeg";
            }
        }
        return new FileInfo(name, contentType, size);
    }

    public void health(JsonCallback callback) {
        submit(() -> {
            try {
                return requestJson("GET", "/v1/upstream-health");
            } catch (ApiError error) {
                if (error.status == 404) {
                    return requestJson("GET", "/health");
                }
                throw error;
            }
        }, callback);
    }

    public void listPets(JsonCallback callback) {
        submit(() -> requestJson("GET", "/v1/pets"), callback);
    }

    public void getPet(String petId, JsonCallback callback) {
        submit(() -> requestJson("GET", "/v1/pets/" + Uri.encode(petId)), callback);
    }

    public void updatePet(String petId, String displayName, JsonCallback callback) {
        submit(() -> {
            JSONObject body = new JSONObject();
            body.put("display_name", displayName);
            return requestJson(
                    "PATCH",
                    "/v1/pets/" + Uri.encode(petId),
                    body
            );
        }, callback);
    }

    public void deletePet(String petId, JsonCallback callback) {
        submit(() -> requestJson(
                "DELETE",
                "/v1/pets/" + Uri.encode(petId)
        ), callback);
    }

    public void deleteImage(String petId, String imageId, JsonCallback callback) {
        submit(() -> requestJson(
                "DELETE",
                "/v1/pets/" + Uri.encode(petId)
                        + "/images/" + Uri.encode(imageId)
        ), callback);
    }

    public void identify(Uri image, JsonCallback callback) {
        submit(() -> uploadImage(
                "/v1/identify?top_k=5",
                "file",
                image,
                null,
                null
        ), callback);
    }

    public void enroll(
            String petId,
            String displayName,
            Uri image,
            JsonCallback callback
    ) {
        enroll(petId, displayName, java.util.Collections.singletonList(image), callback);
    }

    public void enroll(
            String petId,
            String displayName,
            List<Uri> images,
            JsonCallback callback
    ) {
        submit(() -> uploadImages(
                "/v1/pets/" + Uri.encode(petId) + "/images",
                "files",
                images,
                "display_name",
                displayName
        ), callback);
    }

    private void submit(JsonRequest request, JsonCallback callback) {
        if (closed) {
            callback.onError(new ApiError("Client is closed", 0, "client_closed", null));
            return;
        }
        executor.execute(() -> {
            try {
                JSONObject result = request.run();
                deliver(() -> callback.onSuccess(result));
            } catch (ApiError error) {
                deliver(() -> callback.onError(error));
            } catch (Exception error) {
                ApiError wrapped = new ApiError(
                        error.getMessage() == null ? "Network request failed" : error.getMessage(),
                        0,
                        "network_error",
                        error
                );
                deliver(() -> callback.onError(wrapped));
            }
        });
    }

    private void deliver(Runnable callback) {
        mainHandler.post(() -> {
            if (!closed) {
                callback.run();
            }
        });
    }

    private JSONObject requestJson(String method, String path) throws ApiError {
        return requestJson(method, path, null);
    }

    private JSONObject requestJson(String method, String path, JSONObject body) throws ApiError {
        HttpURLConnection connection = null;
        try {
            connection = openConnection(path);
            connection.setRequestMethod(method);
            connection.setRequestProperty("Accept", "application/json");
            if (body != null) {
                byte[] payload = body.toString().getBytes(StandardCharsets.UTF_8);
                connection.setDoOutput(true);
                connection.setFixedLengthStreamingMode(payload.length);
                connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");
                try (OutputStream output = connection.getOutputStream()) {
                    output.write(payload);
                }
            }
            return readJsonResponse(connection);
        } catch (IOException | JSONException error) {
            throw new ApiError(messageOf(error), 0, "network_error", error);
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private JSONObject uploadImage(
            String path,
            String fileField,
            Uri image,
            String textField,
            String textValue
    ) throws ApiError {
        return uploadImages(
                path,
                fileField,
                java.util.Collections.singletonList(image),
                textField,
                textValue
        );
    }

    private JSONObject uploadImages(
            String path,
            String fileField,
            List<Uri> images,
            String textField,
            String textValue
    ) throws ApiError {
        if (images == null || images.isEmpty()) {
            throw new ApiError("At least one image is required", 0, "invalid_request", null);
        }
        String boundary = "PawprintId-" + UUID.randomUUID();
        HttpURLConnection connection = null;
        try {
            connection = openConnection(path);
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.setChunkedStreamingMode(64 * 1024);
            connection.setRequestProperty("Accept", "application/json");
            connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

            try (OutputStream output = connection.getOutputStream()) {
                if (textField != null && textValue != null && !textValue.trim().isEmpty()) {
                    writeUtf8(output, "--" + boundary + "\r\n");
                    writeUtf8(output, "Content-Disposition: form-data; name=\""
                            + safeHeaderToken(textField) + "\"\r\n\r\n");
                    writeUtf8(output, textValue.trim());
                    writeUtf8(output, "\r\n");
                }

                for (Uri image : images) {
                    if (image == null) {
                        continue;
                    }
                    FileInfo info = inspect(image);
                    writeUtf8(output, "--" + boundary + "\r\n");
                    writeUtf8(output, "Content-Disposition: form-data; name=\""
                            + safeHeaderToken(fileField) + "\"; filename=\""
                            + safeFilename(info.name) + "\"\r\n");
                    writeUtf8(output, "Content-Type: " + info.contentType + "\r\n");
                    writeUtf8(output, "Content-Transfer-Encoding: binary\r\n\r\n");
                    try (InputStream input = resolver.openInputStream(image)) {
                        if (input == null) {
                            throw new IOException("The selected image cannot be opened");
                        }
                        byte[] buffer = new byte[64 * 1024];
                        int read;
                        while ((read = input.read(buffer)) >= 0) {
                            if (read > 0) {
                                output.write(buffer, 0, read);
                            }
                        }
                    }
                    writeUtf8(output, "\r\n");
                }
                writeUtf8(output, "--" + boundary + "--\r\n");
            }
            return readJsonResponse(connection);
        } catch (IOException | JSONException error) {
            throw new ApiError(messageOf(error), 0, "network_error", error);
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private HttpURLConnection openConnection(String path) throws IOException {
        if (baseUrl == null || baseUrl.isEmpty()) {
            throw new IOException("Server address is not configured");
        }
        HttpURLConnection connection = (HttpURLConnection) new URL(baseUrl + path).openConnection();
        connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
        connection.setReadTimeout(READ_TIMEOUT_MS);
        connection.setUseCaches(false);
        connection.setInstanceFollowRedirects(false);
        return connection;
    }

    private JSONObject readJsonResponse(HttpURLConnection connection)
            throws IOException, JSONException, ApiError {
        int status = connection.getResponseCode();
        InputStream stream = status >= 200 && status < 300
                ? connection.getInputStream()
                : connection.getErrorStream();
        String body = stream == null ? "" : readUtf8(stream);
        if (status < 200 || status >= 300) {
            String code = "http_" + status;
            String message = "HTTP " + status;
            if (!body.trim().isEmpty()) {
                try {
                    JSONObject envelope = new JSONObject(body);
                    JSONObject error = envelope.optJSONObject("error");
                    if (error != null) {
                        code = error.optString("code", code);
                        message = error.optString("message", message);
                    } else {
                        message = envelope.optString("detail", message);
                    }
                } catch (JSONException ignored) {
                    // Keep the normalized HTTP error when the body is not JSON.
                }
            }
            throw new ApiError(message, status, code, null);
        }
        if (body.trim().isEmpty()) {
            return new JSONObject();
        }
        return new JSONObject(body);
    }

    private static String readUtf8(InputStream input) throws IOException {
        try (InputStream stream = input; ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16 * 1024];
            int total = 0;
            int read;
            while ((read = stream.read(buffer)) >= 0) {
                if (read == 0) {
                    continue;
                }
                total += read;
                if (total > MAX_JSON_BYTES) {
                    throw new IOException("Server response is too large");
                }
                output.write(buffer, 0, read);
            }
            return output.toString(StandardCharsets.UTF_8.name());
        }
    }

    private static void writeUtf8(OutputStream output, String value) throws IOException {
        output.write(value.getBytes(StandardCharsets.UTF_8));
    }

    private static String safeHeaderToken(String value) {
        return value.replace("\r", "").replace("\n", "").replace("\"", "");
    }

    private static String safeFilename(String value) {
        String cleaned = safeHeaderToken(value).replace("\\", "_").replace("/", "_");
        return cleaned.isEmpty() ? "pet-image.jpg" : cleaned;
    }

    private static String messageOf(Exception error) {
        return error.getMessage() == null ? "Network request failed" : error.getMessage();
    }

    @Override
    public void close() {
        closed = true;
        executor.shutdownNow();
        mainHandler.removeCallbacksAndMessages(null);
    }
}
