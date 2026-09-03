package com.pawprintid.mobile;

import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/** Stores a small metadata-only recognition history on the device. */
final class RecentRecognitionStore {
    private static final String KEY_RECENT_RESULTS = "recent_results_v1";
    private static final int MAX_ENTRIES = 10;
    private static final int MAX_CANDIDATES = 3;

    private final SharedPreferences preferences;

    RecentRecognitionStore(SharedPreferences preferences) {
        this.preferences = preferences;
    }

    JSONArray read() {
        String value = preferences.getString(KEY_RECENT_RESULTS, "");
        if (value == null || value.trim().isEmpty()) {
            return new JSONArray();
        }
        try {
            return new JSONArray(value);
        } catch (JSONException ignored) {
            preferences.edit().remove(KEY_RECENT_RESULTS).apply();
            return new JSONArray();
        }
    }

    void add(JSONObject response) {
        JSONObject item = summary(response);
        JSONArray previous = read();
        JSONArray next = new JSONArray();
        next.put(item);
        for (int index = 0; index < previous.length() && next.length() < MAX_ENTRIES; index++) {
            JSONObject older = previous.optJSONObject(index);
            if (older != null) {
                next.put(older);
            }
        }
        preferences.edit().putString(KEY_RECENT_RESULTS, next.toString()).apply();
    }

    void clear() {
        preferences.edit().remove(KEY_RECENT_RESULTS).apply();
    }

    private JSONObject summary(JSONObject response) {
        JSONObject item = new JSONObject();
        try {
            item.put("saved_at_ms", System.currentTimeMillis());
            item.put("accepted", response.optBoolean("accepted", false));
            copy(response, item, "decision");
            copy(response, item, "predicted_pet_id");
            copy(response, item, "predicted_display_name");
            copy(response, item, "top1_score");
            copy(response, item, "margin");

            JSONArray sourceCandidates = response.optJSONArray("candidates");
            JSONArray candidates = new JSONArray();
            if (sourceCandidates != null) {
                int count = Math.min(sourceCandidates.length(), MAX_CANDIDATES);
                for (int index = 0; index < count; index++) {
                    JSONObject source = sourceCandidates.optJSONObject(index);
                    if (source == null) {
                        continue;
                    }
                    JSONObject candidate = new JSONObject();
                    copy(source, candidate, "pet_id");
                    copy(source, candidate, "display_name");
                    copy(source, candidate, "score");
                    copy(source, candidate, "reference_count");
                    candidates.put(candidate);
                }
            }
            item.put("candidates", candidates);
        } catch (JSONException ignored) {
            // All copied values originate from valid server JSON and primitive metadata.
        }
        return item;
    }

    private static void copy(JSONObject source, JSONObject target, String key)
            throws JSONException {
        if (source.has(key) && !source.isNull(key)) {
            target.put(key, source.opt(key));
        }
    }
}
