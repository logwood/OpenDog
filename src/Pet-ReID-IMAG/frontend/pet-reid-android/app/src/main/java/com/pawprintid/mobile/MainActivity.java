package com.pawprintid.mobile;

import android.animation.ValueAnimator;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowInsets;
import android.view.animation.DecelerateInterpolator;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.regex.Pattern;

/**
 * Native Android user interface for Pawprint ID.
 *
 * <p>The model remains on the user's computer and is reached over its JSON API.
 * Every screen and interaction in this activity is an Android View; no WebView
 * or browser engine is used.</p>
 */
public final class MainActivity extends Activity {
    private static final String PREFS_NAME = "pawprint_id";
    private static final String PREF_SERVER_URL = "server_url";
    private static final String STATE_SELECTED_PAGE = "selected_page";
    private static final int PAGE_STATUS = 0;
    private static final int PAGE_IDENTIFY = 1;
    private static final int PAGE_GALLERY = 2;
    private static final int REQUEST_QUERY_GALLERY = 7101;
    private static final int REQUEST_QUERY_CAMERA = 7102;
    private static final int REQUEST_ENROLL_GALLERY = 7103;
    private static final int REQUEST_ENROLL_CAMERA = 7104;
    private static final long MAX_IMAGE_BYTES = 32L * 1024L * 1024L;
    private static final int MAX_ENROLL_IMAGES = 8;
    private static final Pattern PET_ID_PATTERN =
            Pattern.compile("^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$");

    private SharedPreferences preferences;
    private PetReIdApiClient apiClient;
    private ImagePreviewLoader previewLoader;
    private RecentRecognitionStore recentRecognitionStore;
    private String currentServerUrl = "";
    private int endpointRevision;
    private int selectedPage = PAGE_IDENTIFY;
    private boolean identifying;
    private boolean enrolling;
    private boolean galleryMutationInFlight;
    private JSONObject latestRecognition;

    private Uri queryImageUri;
    private final ArrayList<Uri> enrollImageUris = new ArrayList<>();
    private Uri pendingCameraUri;
    private String pendingCameraPath;
    private int pendingCameraRequest = -1;

    private View statusPage;
    private View identifyPage;
    private ScrollView galleryPage;
    private Button navStatus;
    private Button navIdentify;
    private Button navGallery;
    private TextView toolbarConnectionStatus;
    private ProgressBar globalProgress;

    private TextView statusBadge;
    private TextView statusCheckedAt;
    private TextView statusDetail;
    private TextView statusServerAddress;
    private TextView statusPetCount;
    private TextView statusImageCount;
    private TextView statusModelProvider;
    private TextView statusModelArchitecture;
    private TextView statusModelFingerprint;

    private ImageView queryImagePreview;
    private View queryImagePlaceholder;
    private TextView queryFileName;
    private TextView identifyError;
    private Button identifyButton;
    private Button queryCameraButton;
    private Button queryGalleryButton;
    private ProgressBar identifyProgress;
    private View identifyResultCard;
    private TextView resultDecision;
    private TextView resultName;
    private TextView resultPetId;
    private TextView resultScore;
    private TextView resultMargin;
    private TextView resultLatency;
    private LinearLayout resultCandidates;
    private View resultRecommendationPanel;
    private TextView resultRecommendation;
    private Button shareResultButton;
    private Button clearRecentButton;
    private View recentRecognitionEmpty;
    private LinearLayout recentRecognitionContainer;

    private TextView galleryLoading;
    private View galleryEmpty;
    private LinearLayout petsContainer;
    private EditText enrollPetId;
    private EditText enrollDisplayName;
    private ImageView enrollImagePreview;
    private View enrollImagePlaceholder;
    private TextView enrollFileName;
    private TextView enrollError;
    private Button enrollButton;
    private Button enrollCameraButton;
    private Button enrollGalleryButton;
    private ProgressBar enrollProgress;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_main);

        preferences = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        recentRecognitionStore = new RecentRecognitionStore(preferences);
        bindViews();
        configureSystemBars();
        bindActions();
        renderRecentRecognitions();
        removeStaleCameraFiles();

        currentServerUrl = normalizeStoredUrl(preferences.getString(PREF_SERVER_URL, ""));
        if (!currentServerUrl.isEmpty() && validateServerUrl(currentServerUrl) != null) {
            currentServerUrl = "";
        }
        apiClient = new PetReIdApiClient(this, currentServerUrl);
        previewLoader = new ImagePreviewLoader(this);
        statusServerAddress.setText(
                currentServerUrl.isEmpty() ? getString(R.string.status_not_configured) : currentServerUrl
        );
        selectPage(PAGE_IDENTIFY);

        if (currentServerUrl.isEmpty()) {
            setServiceState(ServiceState.NOT_CONFIGURED, getString(R.string.connection_help));
            findViewById(R.id.root).post(() -> showServerDialog(true));
        } else {
            refreshAll();
        }
    }

    private void bindViews() {
        statusPage = findViewById(R.id.status_page);
        identifyPage = findViewById(R.id.identify_page);
        galleryPage = findViewById(R.id.gallery_page);
        navStatus = findViewById(R.id.nav_status);
        navIdentify = findViewById(R.id.nav_identify);
        navGallery = findViewById(R.id.nav_gallery);
        toolbarConnectionStatus = findViewById(R.id.toolbar_connection_status);
        globalProgress = findViewById(R.id.global_progress);

        statusBadge = findViewById(R.id.status_badge);
        statusCheckedAt = findViewById(R.id.status_checked_at);
        statusDetail = findViewById(R.id.status_detail);
        statusServerAddress = findViewById(R.id.status_server_address);
        statusPetCount = findViewById(R.id.status_pet_count);
        statusImageCount = findViewById(R.id.status_image_count);
        statusModelProvider = findViewById(R.id.status_model_provider_value);
        statusModelArchitecture = findViewById(R.id.status_model_architecture_value);
        statusModelFingerprint = findViewById(R.id.status_model_fingerprint_value);

        queryImagePreview = findViewById(R.id.query_image_preview);
        queryImagePlaceholder = findViewById(R.id.query_image_placeholder);
        queryFileName = findViewById(R.id.query_file_name);
        identifyError = findViewById(R.id.identify_error);
        identifyButton = findViewById(R.id.identify_button);
        queryCameraButton = findViewById(R.id.query_camera_button);
        queryGalleryButton = findViewById(R.id.query_gallery_button);
        identifyProgress = findViewById(R.id.identify_progress);
        identifyResultCard = findViewById(R.id.identify_result_card);
        resultDecision = findViewById(R.id.result_decision);
        resultName = findViewById(R.id.result_name);
        resultPetId = findViewById(R.id.result_pet_id);
        resultScore = findViewById(R.id.result_score);
        resultMargin = findViewById(R.id.result_margin);
        resultLatency = findViewById(R.id.result_latency);
        resultCandidates = findViewById(R.id.result_candidates);
        resultRecommendationPanel = findViewById(R.id.result_recommendation_panel);
        resultRecommendation = findViewById(R.id.result_recommendation);
        shareResultButton = findViewById(R.id.share_result_button);
        clearRecentButton = findViewById(R.id.clear_recent_button);
        recentRecognitionEmpty = findViewById(R.id.recent_recognition_empty);
        recentRecognitionContainer = findViewById(R.id.recent_recognition_container);

        galleryLoading = findViewById(R.id.gallery_loading);
        galleryEmpty = findViewById(R.id.gallery_empty);
        petsContainer = findViewById(R.id.pets_container);
        enrollPetId = findViewById(R.id.enroll_pet_id);
        enrollDisplayName = findViewById(R.id.enroll_display_name);
        enrollImagePreview = findViewById(R.id.enroll_image_preview);
        enrollImagePlaceholder = findViewById(R.id.enroll_image_placeholder);
        enrollFileName = findViewById(R.id.enroll_file_name);
        enrollError = findViewById(R.id.enroll_error);
        enrollButton = findViewById(R.id.enroll_button);
        enrollCameraButton = findViewById(R.id.enroll_camera_button);
        enrollGalleryButton = findViewById(R.id.enroll_gallery_button);
        enrollProgress = findViewById(R.id.enroll_progress);
    }

    private void bindActions() {
        navStatus.setOnClickListener(view -> selectPage(PAGE_STATUS));
        navIdentify.setOnClickListener(view -> selectPage(PAGE_IDENTIFY));
        navGallery.setOnClickListener(view -> selectPage(PAGE_GALLERY));
        findViewById(R.id.server_button).setOnClickListener(view -> showServerDialog(false));
        findViewById(R.id.retry_connection_button).setOnClickListener(view -> refreshAll());
        findViewById(R.id.gallery_refresh_button).setOnClickListener(view -> refreshGallery());

        queryCameraButton.setOnClickListener(view -> launchCamera(REQUEST_QUERY_CAMERA));
        queryGalleryButton.setOnClickListener(view -> launchGallery(REQUEST_QUERY_GALLERY));
        identifyButton.setOnClickListener(view -> identifySelectedImage());
        findViewById(R.id.reset_identify_button).setOnClickListener(view -> resetIdentification());
        shareResultButton.setOnClickListener(view -> shareLatestRecognition());
        clearRecentButton.setOnClickListener(view -> confirmClearRecentRecognitions());

        enrollCameraButton.setOnClickListener(view -> launchCamera(REQUEST_ENROLL_CAMERA));
        enrollGalleryButton.setOnClickListener(view -> launchGallery(REQUEST_ENROLL_GALLERY));
        enrollButton.setOnClickListener(view -> enrollSelectedImage());
    }

    private enum ServiceState {
        NOT_CONFIGURED,
        CHECKING,
        ONLINE,
        OFFLINE
    }

    private void configureSystemBars() {
        getWindow().setStatusBarColor(getColor(R.color.pawprint_ink));
        getWindow().setNavigationBarColor(getColor(R.color.pawprint_ink));
        if (Build.VERSION.SDK_INT >= 35) {
            View root = findViewById(R.id.root);
            root.setOnApplyWindowInsetsListener((view, insets) -> {
                android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars());
                view.setPadding(bars.left, bars.top, bars.right, bars.bottom);
                return insets;
            });
            root.requestApplyInsets();
        }
    }

    private void selectPage(int page) {
        View target = page == PAGE_STATUS
                ? statusPage
                : (page == PAGE_GALLERY ? galleryPage : identifyPage);
        boolean entering = target.getVisibility() != View.VISIBLE;
        resetMotionState(statusPage);
        resetMotionState(identifyPage);
        resetMotionState(galleryPage);
        selectedPage = page;
        statusPage.setVisibility(page == PAGE_STATUS ? View.VISIBLE : View.GONE);
        identifyPage.setVisibility(page == PAGE_IDENTIFY ? View.VISIBLE : View.GONE);
        galleryPage.setVisibility(page == PAGE_GALLERY ? View.VISIBLE : View.GONE);
        navStatus.setSelected(page == PAGE_STATUS);
        navIdentify.setSelected(page == PAGE_IDENTIFY);
        navGallery.setSelected(page == PAGE_GALLERY);
        if (entering) {
            animateViewIn(target, 10, 0L);
        }
    }

    private void refreshAll() {
        if (currentServerUrl.isEmpty()) {
            setServiceState(ServiceState.NOT_CONFIGURED, getString(R.string.connection_help));
            showServerDialog(false);
            return;
        }
        final int revision = ++endpointRevision;
        statusServerAddress.setText(currentServerUrl);
        setServiceState(ServiceState.CHECKING, getString(R.string.status_waiting));
        apiClient.health(new PetReIdApiClient.JsonCallback() {
            @Override
            public void onSuccess(JSONObject response) {
                if (revision != endpointRevision) {
                    return;
                }
                renderHealth(response);
                setServiceState(ServiceState.ONLINE, getString(R.string.connection_ok_detail));
                refreshGallery(revision);
            }

            @Override
            public void onError(PetReIdApiClient.ApiError error) {
                if (revision != endpointRevision) {
                    return;
                }
                galleryLoading.setVisibility(View.GONE);
                setServiceState(
                        ServiceState.OFFLINE,
                        getString(
                                R.string.connection_failed_detail,
                                currentServerUrl,
                                friendlyError(error)
                        )
                );
            }
        });
    }

    private void renderHealth(JSONObject response) {
        JSONObject gallery = response.optJSONObject("gallery");
        if (gallery != null) {
            statusPetCount.setText(String.valueOf(gallery.optInt("pets", 0)));
            statusImageCount.setText(String.valueOf(gallery.optInt("reference_images", 0)));
        }
        JSONObject backend = response.optJSONObject("backend");
        String provider = deepString(
                backend,
                "provider",
                "runtime_provider",
                "execution_provider",
                "device",
                "backend"
        );
        String architecture = deepString(
                backend,
                "architecture",
                "model_architecture",
                "model_type",
                "profile",
                "mode"
        );
        String fingerprint = response.optString("model_fingerprint", "");
        statusModelProvider.setText(blankAsDash(provider));
        statusModelArchitecture.setText(blankAsDash(architecture));
        statusModelFingerprint.setText(blankAsDash(fingerprint));
    }

    private void setServiceState(ServiceState state, String detail) {
        int fillColor;
        int textColor;
        String label;
        switch (state) {
            case ONLINE:
                fillColor = getColor(R.color.pawprint_mint);
                textColor = getColor(R.color.pawprint_green_dark);
                label = getString(R.string.status_connected);
                break;
            case CHECKING:
                fillColor = Color.rgb(239, 235, 208);
                textColor = getColor(R.color.pawprint_warning);
                label = getString(R.string.status_connecting);
                break;
            case OFFLINE:
                fillColor = Color.rgb(247, 226, 224);
                textColor = getColor(R.color.pawprint_error);
                label = getString(R.string.status_offline);
                break;
            default:
                fillColor = Color.rgb(229, 234, 229);
                textColor = getColor(R.color.pawprint_muted);
                label = getString(R.string.status_not_configured);
                break;
        }
        stylePill(toolbarConnectionStatus, fillColor, textColor);
        stylePill(statusBadge, fillColor, textColor);
        toolbarConnectionStatus.setText(label);
        statusBadge.setText(label);
        statusDetail.setText(detail);
        globalProgress.setVisibility(state == ServiceState.CHECKING ? View.VISIBLE : View.GONE);
        if (state == ServiceState.CHECKING) {
            statusCheckedAt.setText(R.string.status_waiting);
        } else if (state != ServiceState.NOT_CONFIGURED) {
            String time = DateFormat.getTimeInstance(DateFormat.SHORT, Locale.CHINA)
                    .format(new Date());
            statusCheckedAt.setText(getString(R.string.status_last_checked, time));
        }
    }

    private void stylePill(TextView view, int fillColor, int textColor) {
        GradientDrawable background = new GradientDrawable();
        background.setColor(fillColor);
        background.setCornerRadius(dp(14));
        view.setBackground(background);
        view.setTextColor(textColor);
    }

    private void showServerDialog(boolean firstRun) {
        EditText input = new EditText(this);
        input.setSingleLine(true);
        input.setHint(R.string.server_url_hint);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        input.setText(currentServerUrl);
        input.setSelection(input.length());

        int horizontal = dp(24);
        LinearLayout wrapper = new LinearLayout(this);
        wrapper.setOrientation(LinearLayout.VERTICAL);
        wrapper.setPadding(horizontal, 0, horizontal, 0);
        wrapper.addView(input, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle(R.string.server_dialog_title)
                .setMessage(R.string.server_dialog_message)
                .setView(wrapper)
                .setNegativeButton(R.string.server_dialog_cancel, null)
                .setPositiveButton(R.string.server_dialog_save, null)
                .create();
        dialog.setOnShowListener(ignored -> dialog
                .getButton(DialogInterface.BUTTON_POSITIVE)
                .setOnClickListener(view -> {
                    String validation = validateServerUrl(input.getText().toString());
                    if (validation != null) {
                        input.setError(validation);
                        return;
                    }
                    currentServerUrl = normalizeStoredUrl(input.getText().toString());
                    preferences.edit().putString(PREF_SERVER_URL, currentServerUrl).apply();
                    apiClient.setBaseUrl(currentServerUrl);
                    statusServerAddress.setText(currentServerUrl);
                    dialog.dismiss();
                    refreshAll();
                }));
        dialog.setOnCancelListener(ignored -> {
            if (firstRun && currentServerUrl.isEmpty()) {
                setServiceState(
                        ServiceState.NOT_CONFIGURED,
                        getString(R.string.connection_help)
                );
            }
        });
        dialog.show();
        input.requestFocus();
    }

    private void launchGallery(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        intent.putExtra(
                Intent.EXTRA_ALLOW_MULTIPLE,
                requestCode == REQUEST_ENROLL_GALLERY
        );
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        try {
            startActivityForResult(
                    Intent.createChooser(intent, getString(R.string.image_picker_title)),
                    requestCode
            );
        } catch (ActivityNotFoundException error) {
            Toast.makeText(this, R.string.file_picker_unavailable, Toast.LENGTH_LONG).show();
        }
    }

    private void launchCamera(int requestCode) {
        if (!getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
            Toast.makeText(this, R.string.file_picker_unavailable, Toast.LENGTH_LONG).show();
            return;
        }
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) == null) {
            Toast.makeText(this, R.string.file_picker_unavailable, Toast.LENGTH_LONG).show();
            return;
        }
        File directory = new File(getCacheDir(), "camera");
        if (!directory.exists() && !directory.mkdirs()) {
            Toast.makeText(this, R.string.file_picker_unavailable, Toast.LENGTH_LONG).show();
            return;
        }
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                .format(new Date());
        File output = new File(
                directory,
                "PawprintID_" + timestamp + "_" + UUID.randomUUID() + ".jpg"
        );
        pendingCameraPath = output.getAbsolutePath();
        pendingCameraUri = CaptureContentProvider.uriForFile(this, output);
        pendingCameraRequest = requestCode;

        intent.putExtra(MediaStore.EXTRA_OUTPUT, pendingCameraUri);
        intent.setClipData(ClipData.newRawUri(
                getString(R.string.camera_capture_label),
                pendingCameraUri
        ));
        intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                | Intent.FLAG_GRANT_READ_URI_PERMISSION);
        for (ResolveInfo handler : getPackageManager().queryIntentActivities(
                intent,
                PackageManager.MATCH_DEFAULT_ONLY
        )) {
            grantUriPermission(
                    handler.activityInfo.packageName,
                    pendingCameraUri,
                    Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                            | Intent.FLAG_GRANT_READ_URI_PERMISSION
            );
        }
        try {
            startActivityForResult(intent, requestCode);
        } catch (ActivityNotFoundException error) {
            clearPendingCamera(true);
            Toast.makeText(this, R.string.file_picker_unavailable, Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_QUERY_GALLERY || requestCode == REQUEST_ENROLL_GALLERY) {
            if (resultCode != RESULT_OK || data == null) {
                return;
            }
            List<Uri> selected = resultUris(data);
            if (selected.isEmpty()) {
                return;
            }
            if ((data.getFlags() & Intent.FLAG_GRANT_READ_URI_PERMISSION) != 0) {
                for (Uri uri : selected) {
                    try {
                        getContentResolver().takePersistableUriPermission(
                                uri,
                                Intent.FLAG_GRANT_READ_URI_PERMISSION
                        );
                    } catch (SecurityException ignored) {
                        // The temporary activity grant is sufficient for this session.
                    }
                }
            }
            if (requestCode == REQUEST_QUERY_GALLERY) {
                setQueryImage(selected.get(0));
            } else {
                setEnrollImages(selected);
            }
            return;
        }

        if (requestCode != REQUEST_QUERY_CAMERA && requestCode != REQUEST_ENROLL_CAMERA) {
            return;
        }
        Uri captured = null;
        if (requestCode == pendingCameraRequest
                && resultCode == RESULT_OK
                && pendingCameraUri != null
                && pendingCameraPath != null) {
            File file = new File(pendingCameraPath);
            if (file.isFile() && file.length() > 0L) {
                captured = pendingCameraUri;
            }
        }
        if (captured != null) {
            if (requestCode == REQUEST_QUERY_CAMERA) {
                setQueryImage(captured);
            } else {
                setEnrollImages(java.util.Collections.singletonList(captured));
            }
            clearPendingCamera(false);
        } else {
            clearPendingCamera(true);
        }
    }

    private List<Uri> resultUris(Intent data) {
        ArrayList<Uri> result = new ArrayList<>();
        if (data.getClipData() != null) {
            for (int index = 0; index < data.getClipData().getItemCount(); index++) {
                Uri uri = data.getClipData().getItemAt(index).getUri();
                if (uri != null && !result.contains(uri)) {
                    result.add(uri);
                }
            }
        }
        if (result.isEmpty() && data.getData() != null) {
            result.add(data.getData());
        }
        return result;
    }

    private void setQueryImage(Uri uri) {
        PetReIdApiClient.FileInfo info = apiClient.inspect(uri);
        if (info.size > MAX_IMAGE_BYTES) {
            showText(identifyError, getString(R.string.image_too_large));
            return;
        }
        queryImageUri = uri;
        queryFileName.setText(fileDescription(info));
        queryImagePreview.setVisibility(View.VISIBLE);
        queryImagePlaceholder.setVisibility(View.GONE);
        identifyResultCard.setVisibility(View.GONE);
        latestRecognition = null;
        shareResultButton.setEnabled(false);
        hideText(identifyError);
        identifyButton.setEnabled(!identifying);
        previewLoader.load(uri, queryImagePreview, () -> {
            if (uri.equals(queryImageUri)) {
                queryImagePreview.setVisibility(View.GONE);
                queryImagePlaceholder.setVisibility(View.VISIBLE);
                showText(identifyError, getString(R.string.image_read_failed));
            }
        });
    }

    private void setEnrollImages(List<Uri> uris) {
        if (uris == null || uris.isEmpty()) {
            return;
        }
        ArrayList<Uri> valid = new ArrayList<>();
        boolean selectionIssue = uris.size() > MAX_ENROLL_IMAGES;
        if (selectionIssue) {
            showText(enrollError, getString(R.string.enroll_too_many_images));
        }
        int limit = Math.min(uris.size(), MAX_ENROLL_IMAGES);
        for (int index = 0; index < limit; index++) {
            Uri uri = uris.get(index);
            PetReIdApiClient.FileInfo info = apiClient.inspect(uri);
            if (info.size > MAX_IMAGE_BYTES) {
                selectionIssue = true;
                showText(enrollError, getString(R.string.image_too_large));
                continue;
            }
            valid.add(uri);
        }
        if (valid.isEmpty()) {
            return;
        }
        enrollImageUris.clear();
        enrollImageUris.addAll(valid);
        PetReIdApiClient.FileInfo firstInfo = apiClient.inspect(valid.get(0));
        enrollFileName.setText(
                valid.size() == 1
                        ? fileDescription(firstInfo)
                        : getString(R.string.enroll_images_selected, valid.size())
        );
        enrollImagePreview.setVisibility(View.VISIBLE);
        enrollImagePlaceholder.setVisibility(View.GONE);
        if (!selectionIssue) {
            hideText(enrollError);
        }
        Uri previewUri = valid.get(0);
        previewLoader.load(previewUri, enrollImagePreview, () -> {
            if (!enrollImageUris.isEmpty() && previewUri.equals(enrollImageUris.get(0))) {
                enrollImagePreview.setVisibility(View.GONE);
                enrollImagePlaceholder.setVisibility(View.VISIBLE);
                showText(enrollError, getString(R.string.image_read_failed));
            }
        });
    }

    private void resetIdentification() {
        queryImageUri = null;
        previewLoader.clear(queryImagePreview);
        queryImagePreview.setVisibility(View.GONE);
        queryImagePlaceholder.setVisibility(View.VISIBLE);
        queryFileName.setText(R.string.identify_no_image);
        identifyResultCard.setVisibility(View.GONE);
        latestRecognition = null;
        shareResultButton.setEnabled(false);
        identifyButton.setEnabled(false);
        hideText(identifyError);
    }

    private void clearEnrollImage() {
        enrollImageUris.clear();
        previewLoader.clear(enrollImagePreview);
        enrollImagePreview.setVisibility(View.GONE);
        enrollImagePlaceholder.setVisibility(View.VISIBLE);
        enrollFileName.setText(R.string.enroll_image_empty);
    }

    private void clearPendingCamera(boolean deleteFile) {
        if (pendingCameraUri != null) {
            revokeUriPermission(
                    pendingCameraUri,
                    Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION
            );
        }
        if (deleteFile && pendingCameraPath != null) {
            File file = new File(pendingCameraPath);
            if (file.isFile()) {
                file.delete();
            }
        }
        pendingCameraUri = null;
        pendingCameraPath = null;
        pendingCameraRequest = -1;
    }

    private void removeStaleCameraFiles() {
        File directory = new File(getCacheDir(), "camera");
        File[] files = directory.listFiles();
        if (files == null) {
            return;
        }
        long cutoff = System.currentTimeMillis() - 24L * 60L * 60L * 1000L;
        for (File file : files) {
            if (file.isFile() && file.lastModified() < cutoff) {
                file.delete();
            }
        }
    }

    private void identifySelectedImage() {
        if (identifying) {
            return;
        }
        if (queryImageUri == null) {
            showText(identifyError, getString(R.string.identify_no_image));
            return;
        }
        if (currentServerUrl.isEmpty()) {
            showServerDialog(false);
            return;
        }
        setIdentifyBusy(true);
        hideText(identifyError);
        identifyResultCard.setVisibility(View.GONE);
        Uri submitted = queryImageUri;
        apiClient.identify(submitted, new PetReIdApiClient.JsonCallback() {
            @Override
            public void onSuccess(JSONObject response) {
                setIdentifyBusy(false);
                setServiceState(ServiceState.ONLINE, getString(R.string.connection_ok_detail));
                renderIdentification(response);
            }

            @Override
            public void onError(PetReIdApiClient.ApiError error) {
                setIdentifyBusy(false);
                showText(identifyError, friendlyError(error));
                if (isConnectivityError(error)) {
                    setServiceState(ServiceState.OFFLINE, friendlyError(error));
                }
            }
        });
    }

    private void setIdentifyBusy(boolean busy) {
        identifying = busy;
        identifyProgress.setVisibility(busy ? View.VISIBLE : View.GONE);
        identifyButton.setEnabled(!busy && queryImageUri != null);
        identifyButton.setText(busy ? R.string.action_identifying : R.string.action_identify);
        queryCameraButton.setEnabled(!busy);
        queryGalleryButton.setEnabled(!busy);
    }

    private void renderIdentification(JSONObject response) {
        latestRecognition = response;
        shareResultButton.setEnabled(true);
        boolean accepted = response.optBoolean("accepted", false);
        String decision = response.optString("decision", "");
        String petId = nullableString(response, "predicted_pet_id");
        String displayName = nullableString(response, "predicted_display_name");

        resultDecision.setText(decisionLabel(accepted, decision));
        if (accepted) {
            stylePill(
                    resultDecision,
                    getColor(R.color.pawprint_mint),
                    getColor(R.color.pawprint_green_dark)
            );
        } else {
            stylePill(
                    resultDecision,
                    Color.rgb(247, 235, 214),
                    getColor(R.color.pawprint_warning)
            );
        }
        resultName.setText(
                displayName == null || displayName.isEmpty()
                        ? getString(R.string.identify_unconfirmed)
                        : displayName
        );
        resultPetId.setText(
                petId == null || petId.isEmpty() ? getString(R.string.status_unknown) : petId
        );
        resultScore.setText(jsonNumber(response, "top1_score", 4, ""));
        resultMargin.setText(jsonNumber(response, "margin", 4, ""));
        resultLatency.setText(jsonNumber(response, "latency_ms", 0, " ms"));

        JSONObject snapshot = response.optJSONObject("gallery_snapshot");
        if (snapshot != null) {
            statusPetCount.setText(String.valueOf(snapshot.optInt("pets", 0)));
            statusImageCount.setText(String.valueOf(snapshot.optInt("reference_images", 0)));
        }

        resultCandidates.removeAllViews();
        JSONArray candidates = response.optJSONArray("candidates");
        if (candidates == null || candidates.length() == 0) {
            TextView empty = bodyText(getString(R.string.identify_no_candidates));
            empty.setPadding(0, dp(8), 0, dp(8));
            resultCandidates.addView(empty);
        } else {
            for (int index = 0; index < candidates.length(); index++) {
                JSONObject candidate = candidates.optJSONObject(index);
                if (candidate != null) {
                    resultCandidates.addView(candidateRow(candidate, index + 1));
                }
            }
        }

        String recommendation = recommendationText(response);
        if (recommendation.isEmpty()) {
            resultRecommendationPanel.setVisibility(View.GONE);
        } else {
            resultRecommendation.setText(recommendation);
            resultRecommendationPanel.setVisibility(View.VISIBLE);
        }
        identifyResultCard.setVisibility(View.VISIBLE);
        identifyResultCard.requestFocus();
        animateViewIn(identifyResultCard, 14, 0L);
        recentRecognitionStore.add(response);
        renderRecentRecognitions();
    }

    private View candidateRow(JSONObject candidate, int rank) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(12), dp(11), dp(12), dp(11));
        row.setBackgroundResource(R.drawable.soft_card_background);
        LinearLayout.LayoutParams rowParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        rowParams.bottomMargin = dp(7);
        row.setLayoutParams(rowParams);

        TextView rankView = new TextView(this);
        rankView.setText(String.format(Locale.CHINA, "#%d", rank));
        rankView.setTextColor(getColor(R.color.pawprint_green));
        rankView.setTextSize(14);
        rankView.setTypeface(null, android.graphics.Typeface.BOLD);
        row.addView(rankView, new LinearLayout.LayoutParams(dp(42), dp(40)));

        LinearLayout identity = new LinearLayout(this);
        identity.setOrientation(LinearLayout.VERTICAL);
        String name = candidate.optString("display_name", candidate.optString("pet_id", "—"));
        TextView nameView = new TextView(this);
        nameView.setText(name);
        nameView.setTextColor(getColor(R.color.pawprint_text));
        nameView.setTextSize(15);
        nameView.setTypeface(null, android.graphics.Typeface.BOLD);
        identity.addView(nameView);

        String petId = candidate.optString("pet_id", "—");
        int references = candidate.optInt("reference_count", 0);
        TextView idView = new TextView(this);
        idView.setText(String.format(Locale.CHINA, "%s · %d 张参考图", petId, references));
        idView.setTextColor(getColor(R.color.pawprint_muted));
        idView.setTextSize(11);
        identity.addView(idView);
        row.addView(identity, new LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f
        ));

        TextView score = new TextView(this);
        score.setText(jsonNumber(candidate, "score", 4, ""));
        score.setTextColor(getColor(R.color.pawprint_ink));
        score.setTextSize(15);
        score.setTypeface(null, android.graphics.Typeface.BOLD);
        score.setGravity(Gravity.END | Gravity.CENTER_VERTICAL);
        row.addView(score, new LinearLayout.LayoutParams(dp(74), dp(44)));
        return row;
    }

    private void shareLatestRecognition() {
        if (latestRecognition == null) {
            Toast.makeText(this, R.string.share_no_result, Toast.LENGTH_SHORT).show();
            return;
        }
        Intent share = new Intent(Intent.ACTION_SEND);
        share.setType("text/plain");
        share.putExtra(Intent.EXTRA_SUBJECT, getString(R.string.share_result_subject));
        share.putExtra(Intent.EXTRA_TEXT, recognitionShareText(latestRecognition));
        try {
            startActivity(Intent.createChooser(
                    share,
                    getString(R.string.share_chooser_title)
            ));
        } catch (ActivityNotFoundException error) {
            Toast.makeText(this, R.string.share_unavailable, Toast.LENGTH_LONG).show();
        }
    }

    private String recognitionShareText(JSONObject response) {
        boolean accepted = response.optBoolean("accepted", false);
        String decision = decisionLabel(accepted, response.optString("decision", ""));
        String displayName = nullableString(response, "predicted_display_name");
        String petId = nullableString(response, "predicted_pet_id");
        StringBuilder text = new StringBuilder();
        text.append(getString(R.string.share_result_header)).append("\n");
        text.append(getString(R.string.share_result_decision, decision)).append("\n");
        text.append(getString(
                R.string.share_result_identity,
                displayName == null
                        ? getString(R.string.identify_unconfirmed)
                        : displayName
        )).append("\n");
        text.append(getString(
                R.string.share_result_pet_id,
                petId == null ? getString(R.string.status_unknown) : petId
        )).append("\n");
        text.append(getString(
                R.string.share_result_score,
                jsonNumber(response, "top1_score", 4, "")
        )).append("\n");
        text.append(getString(
                R.string.share_result_margin,
                jsonNumber(response, "margin", 4, "")
        ));

        JSONArray candidates = response.optJSONArray("candidates");
        if (candidates != null && candidates.length() > 0) {
            text.append("\n\n").append(getString(R.string.share_result_candidates));
            int count = Math.min(candidates.length(), 3);
            for (int index = 0; index < count; index++) {
                JSONObject candidate = candidates.optJSONObject(index);
                if (candidate == null) {
                    continue;
                }
                String candidatePetId = candidate.optString(
                        "pet_id",
                        getString(R.string.status_unknown)
                );
                String candidateName = candidate.optString("display_name", candidatePetId);
                text.append("\n").append(getString(
                        R.string.share_result_candidate,
                        index + 1,
                        candidateName,
                        candidatePetId,
                        jsonNumber(candidate, "score", 4, "")
                ));
            }
        }
        text.append("\n\n").append(getString(R.string.share_result_note));
        return text.toString();
    }

    private void renderRecentRecognitions() {
        JSONArray items = recentRecognitionStore.read();
        recentRecognitionContainer.removeAllViews();
        int count = items.length();
        recentRecognitionEmpty.setVisibility(count == 0 ? View.VISIBLE : View.GONE);
        clearRecentButton.setEnabled(count > 0);
        for (int index = 0; index < count; index++) {
            JSONObject item = items.optJSONObject(index);
            if (item != null) {
                View row = recentRecognitionRow(item);
                recentRecognitionContainer.addView(row);
                animateViewIn(row, 8, Math.min(index, 6) * 35L);
            }
        }
    }

    private View recentRecognitionRow(JSONObject item) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.VERTICAL);
        row.setPadding(dp(13), dp(12), dp(13), dp(12));
        row.setBackgroundResource(R.drawable.soft_card_background);
        LinearLayout.LayoutParams rowParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        rowParams.topMargin = dp(7);
        row.setLayoutParams(rowParams);

        boolean accepted = item.optBoolean("accepted", false);
        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);

        TextView decision = new TextView(this);
        decision.setText(decisionLabel(accepted, item.optString("decision", "")));
        decision.setTextSize(11);
        decision.setTypeface(null, android.graphics.Typeface.BOLD);
        decision.setPadding(dp(10), dp(5), dp(10), dp(5));
        if (accepted) {
            stylePill(
                    decision,
                    getColor(R.color.pawprint_mint),
                    getColor(R.color.pawprint_green_dark)
            );
        } else {
            stylePill(
                    decision,
                    Color.rgb(247, 235, 214),
                    getColor(R.color.pawprint_warning)
            );
        }
        header.addView(decision);

        TextView time = bodyText(formatRecentTime(item.optLong("saved_at_ms", 0L)));
        time.setTextSize(11);
        time.setGravity(Gravity.END);
        header.addView(time, new LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f
        ));
        row.addView(header);

        String petId = nullableString(item, "predicted_pet_id");
        String displayName = nullableString(item, "predicted_display_name");
        TextView name = new TextView(this);
        name.setText(
                displayName == null
                        ? getString(R.string.identify_unconfirmed)
                        : displayName
        );
        name.setTextColor(getColor(R.color.pawprint_ink));
        name.setTextSize(17);
        name.setTypeface(null, android.graphics.Typeface.BOLD);
        name.setPadding(0, dp(10), 0, 0);
        row.addView(name);

        TextView id = bodyText(
                petId == null ? getString(R.string.status_unknown) : petId
        );
        id.setTextSize(11);
        id.setTypeface(android.graphics.Typeface.MONOSPACE);
        id.setPadding(0, dp(2), 0, 0);
        row.addView(id);

        TextView metrics = bodyText(getString(
                R.string.recent_recognition_metrics,
                jsonNumber(item, "top1_score", 4, ""),
                jsonNumber(item, "margin", 4, "")
        ));
        metrics.setPadding(0, dp(7), 0, 0);
        row.addView(metrics);

        JSONArray candidates = item.optJSONArray("candidates");
        if (candidates != null && candidates.length() > 0) {
            StringBuilder values = new StringBuilder();
            for (int index = 0; index < candidates.length(); index++) {
                JSONObject candidate = candidates.optJSONObject(index);
                if (candidate == null) {
                    continue;
                }
                if (values.length() > 0) {
                    values.append(" · ");
                }
                String candidatePetId = candidate.optString("pet_id", "—");
                String candidateName = candidate.optString("display_name", candidatePetId);
                values.append(index + 1)
                        .append(". ")
                        .append(candidateName)
                        .append(" ")
                        .append(jsonNumber(candidate, "score", 4, ""));
            }
            if (values.length() > 0) {
                TextView candidateText = bodyText(getString(
                        R.string.recent_recognition_candidates,
                        values.toString()
                ));
                candidateText.setTextSize(11);
                candidateText.setPadding(0, dp(5), 0, 0);
                row.addView(candidateText);
            }
        }
        return row;
    }

    private String formatRecentTime(long savedAtMillis) {
        if (savedAtMillis <= 0L) {
            return getString(R.string.status_unknown);
        }
        return new SimpleDateFormat("MM-dd HH:mm", Locale.CHINA)
                .format(new Date(savedAtMillis));
    }

    private void confirmClearRecentRecognitions() {
        if (recentRecognitionStore.read().length() == 0) {
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle(R.string.recent_recognition_clear_title)
                .setMessage(R.string.recent_recognition_clear_message)
                .setNegativeButton(R.string.action_cancel, null)
                .setPositiveButton(R.string.action_clear_recent, (dialog, which) -> {
                    recentRecognitionStore.clear();
                    renderRecentRecognitions();
                    Toast.makeText(
                            MainActivity.this,
                            R.string.recent_recognition_cleared,
                            Toast.LENGTH_SHORT
                    ).show();
                })
                .show();
    }

    private String recommendationText(JSONObject response) {
        JSONObject agent = response.optJSONObject("agent");
        JSONArray recommendations = agent == null
                ? null
                : agent.optJSONArray("capture_recommendations");
        if (recommendations != null && recommendations.length() > 0) {
            StringBuilder text = new StringBuilder();
            for (int index = 0; index < recommendations.length(); index++) {
                String value = recommendations.optString(index, "").trim();
                if (!value.isEmpty()) {
                    if (text.length() > 0) {
                        text.append("\n");
                    }
                    text.append("• ").append(value);
                }
            }
            return text.toString();
        }

        JSONArray reasons = response.optJSONArray("hard_case_reasons");
        if (reasons == null || reasons.length() == 0) {
            return "";
        }
        StringBuilder text = new StringBuilder();
        for (int index = 0; index < reasons.length(); index++) {
            String value = reasonLabel(reasons.optString(index, ""));
            if (!value.isEmpty()) {
                if (text.length() > 0) {
                    text.append("\n");
                }
                text.append("• ").append(value);
            }
        }
        return text.toString();
    }

    private String decisionLabel(boolean accepted, String decision) {
        if (accepted || "matched".equals(decision)) {
            return getString(R.string.identify_matched);
        }
        if ("possible_unknown".equals(decision)) {
            return "可能是图库外身份";
        }
        if ("needs_more_evidence".equals(decision)) {
            return "需要更多证据";
        }
        return getString(R.string.identify_unconfirmed);
    }

    private String reasonLabel(String reason) {
        switch (reason) {
            case "rejected":
                return "当前图片未达到确认阈值，建议补拍。";
            case "low_margin":
            case "low_fused_margin":
                return "前两名差值较小，请换角度再拍一张。";
            case "branch_conflict":
            case "expert_conflict":
                return "不同识别线索存在冲突，请确保画面清晰。";
            case "single_branch":
            case "insufficient_identity_evidence":
                return "身份线索不足，请同时拍清鼻纹、脸部和身体。";
            case "low_quality":
            case "motion_or_focus_blur":
                return "图片清晰度不足，请保持手机稳定并重新对焦。";
            case "uneven_or_extreme_lighting":
                return "光照不均，请在明亮、柔和的环境重新拍摄。";
            case "body_not_detected":
                return "未检测到完整身体，请后退一点重新构图。";
            default:
                return reason.replace('_', ' ');
        }
    }

    private void refreshGallery() {
        if (currentServerUrl.isEmpty()) {
            showServerDialog(false);
            return;
        }
        refreshGallery(endpointRevision);
    }

    private void refreshGallery(int revision) {
        galleryLoading.setText(R.string.gallery_loading);
        galleryLoading.setTextColor(getColor(R.color.pawprint_muted));
        galleryLoading.setVisibility(View.VISIBLE);
        apiClient.listPets(new PetReIdApiClient.JsonCallback() {
            @Override
            public void onSuccess(JSONObject response) {
                if (revision != endpointRevision) {
                    return;
                }
                galleryLoading.setVisibility(View.GONE);
                renderPets(response.optJSONArray("pets"));
            }

            @Override
            public void onError(PetReIdApiClient.ApiError error) {
                if (revision != endpointRevision) {
                    return;
                }
                galleryLoading.setText(friendlyError(error));
                galleryLoading.setTextColor(getColor(R.color.pawprint_error));
                galleryLoading.setVisibility(View.VISIBLE);
                if (isConnectivityError(error)) {
                    setServiceState(ServiceState.OFFLINE, friendlyError(error));
                }
            }
        });
    }

    private void renderPets(JSONArray pets) {
        petsContainer.removeAllViews();
        int count = pets == null ? 0 : pets.length();
        int referenceCount = 0;
        if (pets != null) {
            for (int index = 0; index < pets.length(); index++) {
                JSONObject pet = pets.optJSONObject(index);
                if (pet != null) {
                    referenceCount += pet.optInt("reference_count", 0);
                    View row = petRow(pet, index);
                    petsContainer.addView(row);
                    animateViewIn(row, 8, Math.min(index, 6) * 35L);
                }
            }
        }
        galleryEmpty.setVisibility(count == 0 ? View.VISIBLE : View.GONE);
        statusPetCount.setText(String.valueOf(count));
        statusImageCount.setText(String.valueOf(referenceCount));
    }

    private View petRow(JSONObject pet, int index) {
        String petId = pet.optString("pet_id", "");
        String displayName = pet.optString("display_name", petId);
        int references = pet.optInt("reference_count", 0);

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(14), dp(13), dp(10), dp(13));
        row.setBackgroundResource(R.drawable.card_background);
        row.setElevation(dp(1));
        row.setClickable(true);
        row.setFocusable(true);
        row.setContentDescription(displayName + "，" + references + " 张参考图");
        LinearLayout.LayoutParams rowParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        rowParams.topMargin = dp(7);
        row.setLayoutParams(rowParams);

        TextView avatar = new TextView(this);
        String initial = displayName.trim().isEmpty()
                ? "?"
                : displayName.trim().substring(0, 1).toUpperCase(Locale.CHINA);
        avatar.setText(initial);
        avatar.setGravity(Gravity.CENTER);
        avatar.setTextSize(19);
        avatar.setTypeface(null, android.graphics.Typeface.BOLD);
        avatar.setTextColor(getColor(R.color.pawprint_green_dark));
        GradientDrawable circle = new GradientDrawable();
        circle.setShape(GradientDrawable.OVAL);
        circle.setColor(index % 2 == 0
                ? getColor(R.color.pawprint_mint)
                : Color.rgb(244, 220, 198));
        avatar.setBackground(circle);
        LinearLayout.LayoutParams avatarParams = new LinearLayout.LayoutParams(dp(48), dp(48));
        avatarParams.rightMargin = dp(12);
        row.addView(avatar, avatarParams);

        LinearLayout identity = new LinearLayout(this);
        identity.setOrientation(LinearLayout.VERTICAL);
        TextView name = new TextView(this);
        name.setText(displayName);
        name.setTextColor(getColor(R.color.pawprint_text));
        name.setTextSize(16);
        name.setTypeface(null, android.graphics.Typeface.BOLD);
        identity.addView(name);
        TextView metadata = new TextView(this);
        metadata.setText(getString(R.string.gallery_pet_row_meta, petId, references));
        metadata.setTextColor(getColor(R.color.pawprint_muted));
        metadata.setTextSize(11);
        metadata.setSingleLine(true);
        metadata.setEllipsize(android.text.TextUtils.TruncateAt.END);
        identity.addView(metadata);
        row.addView(identity, new LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f
        ));

        Button details = new Button(this);
        details.setText(R.string.action_details);
        details.setTextSize(12);
        details.setTextColor(getColor(R.color.pawprint_green));
        details.setAllCaps(false);
        details.setBackgroundColor(Color.TRANSPARENT);
        row.addView(details, new LinearLayout.LayoutParams(dp(66), dp(48)));

        View.OnClickListener open = view -> showPetDetails(petId);
        row.setOnClickListener(open);
        details.setOnClickListener(open);
        return row;
    }

    private void showPetDetails(String petId) {
        loadPetDetails(petId, true);
    }

    private void loadPetDetails(String petId, boolean announce) {
        if (announce) {
            Toast.makeText(this, "正在读取宠物详情…", Toast.LENGTH_SHORT).show();
        }
        apiClient.getPet(petId, new PetReIdApiClient.JsonCallback() {
            @Override
            public void onSuccess(JSONObject pet) {
                showPetDetailsDialog(pet);
            }

            @Override
            public void onError(PetReIdApiClient.ApiError error) {
                Toast.makeText(MainActivity.this, friendlyError(error), Toast.LENGTH_LONG).show();
            }
        });
    }

    private void showPetDetailsDialog(JSONObject pet) {
        String petId = pet.optString("pet_id", "");
        String displayName = pet.optString("display_name", petId);
        JSONArray images = pet.optJSONArray("images");
        int imageCount = images == null ? 0 : images.length();
        final AlertDialog[] detailsDialog = new AlertDialog[1];

        ScrollView scroll = new ScrollView(this);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(dp(24), dp(6), dp(24), dp(12));
        scroll.addView(content);

        TextView name = new TextView(this);
        name.setText(displayName);
        name.setTextColor(getColor(R.color.pawprint_ink));
        name.setTextSize(23);
        name.setTypeface(null, android.graphics.Typeface.BOLD);
        content.addView(name);

        TextView id = bodyText(getString(R.string.gallery_pet_id, petId));
        id.setTextIsSelectable(true);
        id.setPadding(0, dp(4), 0, 0);
        content.addView(id);

        TextView updated = bodyText(getString(
                R.string.gallery_updated,
                formatServerDate(pet.optString("updated_at", ""))
        ));
        updated.setPadding(0, dp(2), 0, dp(14));
        content.addView(updated);

        TextView managementTitle = new TextView(this);
        managementTitle.setText(R.string.gallery_manage_title);
        managementTitle.setTextColor(getColor(R.color.pawprint_text));
        managementTitle.setTextSize(15);
        managementTitle.setTypeface(null, android.graphics.Typeface.BOLD);
        content.addView(managementTitle);

        LinearLayout managementActions = new LinearLayout(this);
        managementActions.setOrientation(LinearLayout.HORIZONTAL);
        managementActions.setPadding(0, dp(8), 0, dp(6));

        Button renameButton = new Button(this);
        renameButton.setText(R.string.action_rename_pet);
        renameButton.setAllCaps(false);
        renameButton.setTextColor(getColor(R.color.pawprint_green));
        renameButton.setTextSize(13);
        renameButton.setBackgroundResource(R.drawable.button_secondary);
        LinearLayout.LayoutParams renameParams = new LinearLayout.LayoutParams(
                0,
                dp(46),
                1f
        );
        renameParams.rightMargin = dp(6);
        managementActions.addView(renameButton, renameParams);

        Button deletePetButton = new Button(this);
        deletePetButton.setText(R.string.action_delete_pet);
        deletePetButton.setAllCaps(false);
        deletePetButton.setTextColor(getColor(R.color.pawprint_error));
        deletePetButton.setTextSize(13);
        deletePetButton.setBackgroundResource(R.drawable.button_secondary);
        LinearLayout.LayoutParams deletePetParams = new LinearLayout.LayoutParams(
                0,
                dp(46),
                1f
        );
        deletePetParams.leftMargin = dp(6);
        managementActions.addView(deletePetButton, deletePetParams);
        content.addView(managementActions);

        renameButton.setOnClickListener(view -> showRenamePetDialog(
                petId,
                displayName,
                detailsDialog[0]
        ));
        deletePetButton.setOnClickListener(view -> confirmDeletePet(
                petId,
                displayName,
                imageCount,
                detailsDialog[0]
        ));

        TextView imagesTitle = new TextView(this);
        imagesTitle.setText(getString(R.string.gallery_images_title, imageCount));
        imagesTitle.setTextColor(getColor(R.color.pawprint_text));
        imagesTitle.setTextSize(15);
        imagesTitle.setTypeface(null, android.graphics.Typeface.BOLD);
        imagesTitle.setPadding(0, dp(8), 0, 0);
        content.addView(imagesTitle);

        if (images == null || images.length() == 0) {
            TextView empty = bodyText(getString(R.string.gallery_no_image_details));
            empty.setPadding(0, dp(10), 0, 0);
            content.addView(empty);
        } else {
            for (int index = 0; index < images.length(); index++) {
                JSONObject image = images.optJSONObject(index);
                if (image == null) {
                    continue;
                }
                LinearLayout imageItem = new LinearLayout(this);
                imageItem.setOrientation(LinearLayout.VERTICAL);
                imageItem.setPadding(dp(12), dp(10), dp(12), dp(10));
                imageItem.setBackgroundResource(R.drawable.soft_card_background);
                LinearLayout.LayoutParams itemParams = new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                );
                itemParams.topMargin = dp(8);
                imageItem.setLayoutParams(itemParams);

                String imageId = image.optString("image_id", "");
                String originalFilename = image.optString("original_filename", "image");
                LinearLayout imageHeader = new LinearLayout(this);
                imageHeader.setOrientation(LinearLayout.HORIZONTAL);
                imageHeader.setGravity(Gravity.CENTER_VERTICAL);

                TextView filename = new TextView(this);
                filename.setText(originalFilename);
                filename.setTextColor(getColor(R.color.pawprint_text));
                filename.setTextSize(13);
                filename.setTypeface(null, android.graphics.Typeface.BOLD);
                filename.setSingleLine(true);
                filename.setEllipsize(android.text.TextUtils.TruncateAt.MIDDLE);
                imageHeader.addView(filename, new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1f
                ));

                Button deleteImageButton = new Button(this);
                deleteImageButton.setText(R.string.action_delete_image);
                deleteImageButton.setAllCaps(false);
                deleteImageButton.setTextColor(getColor(R.color.pawprint_error));
                deleteImageButton.setTextSize(12);
                deleteImageButton.setBackgroundColor(Color.TRANSPARENT);
                deleteImageButton.setEnabled(!imageId.isEmpty());
                imageHeader.addView(
                        deleteImageButton,
                        new LinearLayout.LayoutParams(dp(66), dp(44))
                );
                imageItem.addView(imageHeader);

                TextView dimensions = bodyText(getString(
                        R.string.gallery_image_meta,
                        formatBytes(image.optLong("byte_size", -1L)),
                        image.optInt("width", 0),
                        image.optInt("height", 0)
                ));
                dimensions.setTextSize(11);
                dimensions.setPadding(0, dp(3), 0, 0);
                imageItem.addView(dimensions);
                deleteImageButton.setOnClickListener(view -> confirmDeleteImage(
                        petId,
                        displayName,
                        imageId,
                        originalFilename,
                        imageCount,
                        detailsDialog[0]
                ));
                content.addView(imageItem);
            }
        }

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle(R.string.gallery_pet_detail_title)
                .setView(scroll)
                .setNeutralButton(R.string.action_use_pet, (dialogInterface, which) -> {
                    enrollPetId.setText(petId);
                    enrollDisplayName.setText(displayName);
                    selectPage(PAGE_GALLERY);
                    galleryPage.post(() -> galleryPage.fullScroll(View.FOCUS_DOWN));
                })
                .setPositiveButton(R.string.action_close, null)
                .create();
        detailsDialog[0] = dialog;
        dialog.show();
    }

    private void showRenamePetDialog(
            String petId,
            String currentDisplayName,
            AlertDialog detailsDialog
    ) {
        EditText input = new EditText(this);
        input.setSingleLine(true);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PERSON_NAME);
        input.setText(currentDisplayName);
        input.selectAll();

        LinearLayout wrapper = new LinearLayout(this);
        wrapper.setOrientation(LinearLayout.VERTICAL);
        wrapper.setPadding(dp(24), 0, dp(24), 0);
        wrapper.addView(input, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        AlertDialog renameDialog = new AlertDialog.Builder(this)
                .setTitle(R.string.gallery_rename_title)
                .setMessage(getString(R.string.gallery_rename_message, petId))
                .setView(wrapper)
                .setNegativeButton(R.string.action_cancel, null)
                .setPositiveButton(R.string.action_save, null)
                .create();
        renameDialog.setOnShowListener(ignored -> {
            Button saveButton = renameDialog.getButton(DialogInterface.BUTTON_POSITIVE);
            saveButton.setOnClickListener(view -> {
                String displayName = input.getText().toString().trim();
                if (displayName.isEmpty()) {
                    input.setError(getString(R.string.gallery_name_required));
                    input.requestFocus();
                    return;
                }
                if (displayName.length() > 128) {
                    input.setError(getString(R.string.enroll_name_too_long));
                    input.requestFocus();
                    return;
                }
                if (!beginGalleryMutation()) {
                    return;
                }
                input.setEnabled(false);
                saveButton.setEnabled(false);
                apiClient.updatePet(petId, displayName, new PetReIdApiClient.JsonCallback() {
                    @Override
                    public void onSuccess(JSONObject updatedPet) {
                        endGalleryMutation();
                        renameDialog.dismiss();
                        dismissDialog(detailsDialog);
                        Toast.makeText(
                                MainActivity.this,
                                R.string.gallery_rename_success,
                                Toast.LENGTH_SHORT
                        ).show();
                        setServiceState(
                                ServiceState.ONLINE,
                                getString(R.string.connection_ok_detail)
                        );
                        refreshGallery();
                        showPetDetailsDialog(updatedPet);
                    }

                    @Override
                    public void onError(PetReIdApiClient.ApiError error) {
                        endGalleryMutation();
                        input.setEnabled(true);
                        saveButton.setEnabled(true);
                        Toast.makeText(
                                MainActivity.this,
                                friendlyError(error),
                                Toast.LENGTH_LONG
                        ).show();
                        if (isConnectivityError(error)) {
                            setServiceState(ServiceState.OFFLINE, friendlyError(error));
                        }
                    }
                });
            });
        });
        renameDialog.show();
        input.requestFocus();
    }

    private void confirmDeletePet(
            String petId,
            String displayName,
            int imageCount,
            AlertDialog detailsDialog
    ) {
        new AlertDialog.Builder(this)
                .setTitle(R.string.gallery_delete_pet_title)
                .setMessage(getString(
                        R.string.gallery_delete_pet_message,
                        displayName,
                        imageCount
                ))
                .setNegativeButton(R.string.action_cancel, null)
                .setPositiveButton(R.string.action_delete_pet, (dialog, which) -> {
                    if (!beginGalleryMutation()) {
                        return;
                    }
                    apiClient.deletePet(petId, new PetReIdApiClient.JsonCallback() {
                        @Override
                        public void onSuccess(JSONObject response) {
                            endGalleryMutation();
                            dismissDialog(detailsDialog);
                            int deletedImages = response.optInt("deleted_images", imageCount);
                            Toast.makeText(
                                    MainActivity.this,
                                    getString(
                                            R.string.gallery_delete_pet_success,
                                            displayName,
                                            deletedImages
                                    ),
                                    Toast.LENGTH_LONG
                            ).show();
                            setServiceState(
                                    ServiceState.ONLINE,
                                    getString(R.string.connection_ok_detail)
                            );
                            refreshGallery();
                        }

                        @Override
                        public void onError(PetReIdApiClient.ApiError error) {
                            endGalleryMutation();
                            Toast.makeText(
                                    MainActivity.this,
                                    friendlyError(error),
                                    Toast.LENGTH_LONG
                            ).show();
                            if ("not_found".equals(error.code)) {
                                dismissDialog(detailsDialog);
                                refreshGallery();
                            }
                            if (isConnectivityError(error)) {
                                setServiceState(ServiceState.OFFLINE, friendlyError(error));
                            }
                        }
                    });
                })
                .show();
    }

    private void confirmDeleteImage(
            String petId,
            String displayName,
            String imageId,
            String originalFilename,
            int imageCount,
            AlertDialog detailsDialog
    ) {
        boolean lastImage = imageCount <= 1;
        new AlertDialog.Builder(this)
                .setTitle(R.string.gallery_delete_image_title)
                .setMessage(lastImage
                        ? getString(
                                R.string.gallery_delete_last_image_message,
                                displayName
                        )
                        : getString(
                                R.string.gallery_delete_image_message,
                                originalFilename
                        ))
                .setNegativeButton(R.string.action_cancel, null)
                .setPositiveButton(R.string.action_delete_image, (dialog, which) -> {
                    if (!beginGalleryMutation()) {
                        return;
                    }
                    apiClient.deleteImage(
                            petId,
                            imageId,
                            new PetReIdApiClient.JsonCallback() {
                        @Override
                        public void onSuccess(JSONObject response) {
                            endGalleryMutation();
                            dismissDialog(detailsDialog);
                            boolean petDeleted = response.optBoolean(
                                    "pet_deleted",
                                    lastImage
                            );
                            Toast.makeText(
                                    MainActivity.this,
                                    petDeleted
                                            ? R.string.gallery_delete_last_image_success
                                            : R.string.gallery_delete_image_success,
                                    Toast.LENGTH_LONG
                            ).show();
                            setServiceState(
                                    ServiceState.ONLINE,
                                    getString(R.string.connection_ok_detail)
                            );
                            refreshGallery();
                            if (!petDeleted) {
                                loadPetDetails(petId, false);
                            }
                        }

                        @Override
                        public void onError(PetReIdApiClient.ApiError error) {
                            endGalleryMutation();
                            Toast.makeText(
                                    MainActivity.this,
                                    friendlyError(error),
                                    Toast.LENGTH_LONG
                            ).show();
                            if ("not_found".equals(error.code)) {
                                dismissDialog(detailsDialog);
                                refreshGallery();
                            }
                            if (isConnectivityError(error)) {
                                setServiceState(ServiceState.OFFLINE, friendlyError(error));
                            }
                        }
                    });
                })
                .show();
    }

    private boolean beginGalleryMutation() {
        if (galleryMutationInFlight) {
            Toast.makeText(
                    this,
                    R.string.gallery_mutation_busy,
                    Toast.LENGTH_SHORT
            ).show();
            return false;
        }
        galleryMutationInFlight = true;
        return true;
    }

    private void endGalleryMutation() {
        galleryMutationInFlight = false;
    }

    private void dismissDialog(AlertDialog dialog) {
        if (dialog != null && dialog.isShowing()) {
            dialog.dismiss();
        }
    }

    private void enrollSelectedImage() {
        if (enrolling) {
            return;
        }
        String petId = enrollPetId.getText().toString().trim();
        String displayName = enrollDisplayName.getText().toString().trim();
        if (petId.isEmpty()) {
            enrollPetId.setError(getString(R.string.enroll_pet_id_required));
            enrollPetId.requestFocus();
            return;
        }
        if (!PET_ID_PATTERN.matcher(petId).matches()) {
            enrollPetId.setError(getString(R.string.enroll_pet_id_invalid));
            enrollPetId.requestFocus();
            return;
        }
        if (displayName.length() > 128) {
            enrollDisplayName.setError(getString(R.string.enroll_name_too_long));
            enrollDisplayName.requestFocus();
            return;
        }
        if (enrollImageUris.isEmpty()) {
            showText(enrollError, getString(R.string.enroll_image_required));
            return;
        }
        if (currentServerUrl.isEmpty()) {
            showServerDialog(false);
            return;
        }

        setEnrollBusy(true);
        hideText(enrollError);
        apiClient.enroll(
                petId,
                displayName,
                new ArrayList<>(enrollImageUris),
                new PetReIdApiClient.JsonCallback() {
            @Override
            public void onSuccess(JSONObject response) {
                setEnrollBusy(false);
                int added = arrayLength(response.optJSONArray("added_image_ids"));
                int duplicates = arrayLength(response.optJSONArray("duplicate_image_ids"));
                Toast.makeText(
                        MainActivity.this,
                        getString(R.string.enroll_success, added, duplicates),
                        Toast.LENGTH_LONG
                ).show();
                JSONObject pet = response.optJSONObject("pet");
                if (pet != null) {
                    enrollPetId.setText(pet.optString("pet_id", petId));
                    enrollDisplayName.setText(
                            pet.optString("display_name", displayName.isEmpty() ? petId : displayName)
                    );
                }
                clearEnrollImage();
                setServiceState(ServiceState.ONLINE, getString(R.string.connection_ok_detail));
                refreshGallery();
            }

            @Override
            public void onError(PetReIdApiClient.ApiError error) {
                setEnrollBusy(false);
                showText(enrollError, friendlyError(error));
                if (isConnectivityError(error)) {
                    setServiceState(ServiceState.OFFLINE, friendlyError(error));
                }
            }
                }
        );
    }

    private void setEnrollBusy(boolean busy) {
        enrolling = busy;
        enrollProgress.setVisibility(busy ? View.VISIBLE : View.GONE);
        enrollButton.setEnabled(!busy);
        enrollButton.setText(busy ? R.string.action_enrolling : R.string.action_enroll);
        enrollCameraButton.setEnabled(!busy);
        enrollGalleryButton.setEnabled(!busy);
        enrollPetId.setEnabled(!busy);
        enrollDisplayName.setEnabled(!busy);
    }

    private String validateServerUrl(String rawValue) {
        String value = rawValue == null ? "" : rawValue.trim();
        if (value.isEmpty()) {
            return getString(R.string.server_url_required);
        }
        try {
            URI parsed = new URI(value);
            String scheme = parsed.getScheme();
            if (scheme == null
                    || (!"http".equalsIgnoreCase(scheme)
                    && !"https".equalsIgnoreCase(scheme))
                    || parsed.getHost() == null
                    || parsed.getUserInfo() != null) {
                return getString(R.string.server_url_invalid);
            }
            if (parsed.getRawQuery() != null
                    || parsed.getRawFragment() != null
                    || (parsed.getRawPath() != null
                    && !parsed.getRawPath().isEmpty()
                    && !"/".equals(parsed.getRawPath()))) {
                return getString(R.string.server_url_path_not_allowed);
            }
            if ("http".equalsIgnoreCase(scheme) && !isPrivateHttpHost(parsed.getHost())) {
                return getString(R.string.server_url_http_private_only);
            }
        } catch (URISyntaxException error) {
            return getString(R.string.server_url_invalid);
        }
        return null;
    }

    private String normalizeStoredUrl(String rawValue) {
        String value = rawValue == null ? "" : rawValue.trim();
        while (value.endsWith("/")) {
            value = value.substring(0, value.length() - 1);
        }
        return value;
    }

    private boolean isPrivateHttpHost(String host) {
        String normalized = host == null ? "" : host.toLowerCase(Locale.US);
        if (normalized.startsWith("[") && normalized.endsWith("]")) {
            normalized = normalized.substring(1, normalized.length() - 1);
        }
        if ("localhost".equals(normalized) || normalized.endsWith(".local")) {
            return true;
        }
        String[] octets = normalized.split("\\.", -1);
        if (octets.length == 4) {
            int[] values = new int[4];
            for (int index = 0; index < octets.length; index++) {
                if (!octets[index].matches("[0-9]{1,3}")) {
                    return false;
                }
                try {
                    values[index] = Integer.parseInt(octets[index]);
                } catch (NumberFormatException error) {
                    return false;
                }
                if (values[index] > 255) {
                    return false;
                }
            }
            return values[0] == 10
                    || values[0] == 127
                    || (values[0] == 169 && values[1] == 254)
                    || (values[0] == 172 && values[1] >= 16 && values[1] <= 31)
                    || (values[0] == 192 && values[1] == 168)
                    || (values[0] == 100 && values[1] >= 64 && values[1] <= 127);
        }
        int zone = normalized.indexOf('%');
        if (zone >= 0) {
            normalized = normalized.substring(0, zone);
        }
        return "::1".equals(normalized)
                || normalized.startsWith("fc")
                || normalized.startsWith("fd")
                || normalized.startsWith("fe8")
                || normalized.startsWith("fe9")
                || normalized.startsWith("fea")
                || normalized.startsWith("feb");
    }

    private String friendlyError(PetReIdApiClient.ApiError error) {
        switch (error.code) {
            case "gallery_empty":
                return getString(R.string.error_gallery_empty);
            case "invalid_pet_image":
                return getString(R.string.error_invalid_image);
            case "gallery_conflict":
                return getString(R.string.error_gallery_conflict);
            case "gallery_model_mismatch":
            case "model_mismatch":
                return getString(R.string.error_model_mismatch);
            case "not_found":
                return getString(R.string.error_not_found);
            case "invalid_request":
                return getString(R.string.error_invalid_request);
            case "upstream_unavailable":
                return getString(R.string.error_upstream);
            case "request_too_large":
            case "upload_too_large":
                return getString(R.string.error_too_large);
            default:
                break;
        }
        if (error.status == 0) {
            return getString(R.string.error_network);
        }
        if (error.status == 401 || error.status == 403) {
            return getString(R.string.error_unauthorized);
        }
        if (error.status == 413) {
            return getString(R.string.error_too_large);
        }
        if (error.status == 502 || error.status == 503) {
            return getString(R.string.error_upstream);
        }
        if (error.status > 0) {
            return getString(R.string.error_http, error.status, error.code);
        }
        return getString(R.string.request_failed);
    }

    private boolean isConnectivityError(PetReIdApiClient.ApiError error) {
        return error.status == 0 || error.status == 502 || error.status == 503;
    }

    private String deepString(JSONObject object, String... keys) {
        return deepString(object, keys, 0);
    }

    private String deepString(JSONObject object, String[] keys, int depth) {
        if (object == null || depth > 3) {
            return "";
        }
        for (String key : keys) {
            Object value = object.opt(key);
            if (value instanceof String && !((String) value).trim().isEmpty()) {
                return ((String) value).trim();
            }
        }
        Iterator<String> iterator = object.keys();
        while (iterator.hasNext()) {
            Object value = object.opt(iterator.next());
            if (value instanceof JSONObject) {
                String nested = deepString((JSONObject) value, keys, depth + 1);
                if (!nested.isEmpty()) {
                    return nested;
                }
            }
        }
        return "";
    }

    private String blankAsDash(String value) {
        return value == null || value.trim().isEmpty()
                ? getString(R.string.status_unknown)
                : value.trim();
    }

    private String nullableString(JSONObject object, String key) {
        if (!object.has(key) || object.isNull(key)) {
            return null;
        }
        String value = object.optString(key, "").trim();
        return value.isEmpty() ? null : value;
    }

    private String jsonNumber(JSONObject object, String key, int digits, String suffix) {
        if (!object.has(key) || object.isNull(key)) {
            return getString(R.string.status_unknown);
        }
        double value = object.optDouble(key, Double.NaN);
        if (!Double.isFinite(value)) {
            return getString(R.string.status_unknown);
        }
        String pattern = "%." + digits + "f%s";
        return String.format(Locale.CHINA, pattern, value, suffix);
    }

    private int arrayLength(JSONArray array) {
        return array == null ? 0 : array.length();
    }

    private String fileDescription(PetReIdApiClient.FileInfo info) {
        if (info.size < 0L) {
            return info.name;
        }
        return info.name + " · " + formatBytes(info.size);
    }

    private String formatBytes(long bytes) {
        if (bytes < 0L) {
            return getString(R.string.status_unknown);
        }
        if (bytes < 1024L) {
            return String.format(Locale.CHINA, "%d B", bytes);
        }
        if (bytes < 1024L * 1024L) {
            return String.format(Locale.CHINA, "%.1f KB", bytes / 1024.0);
        }
        return String.format(Locale.CHINA, "%.1f MB", bytes / (1024.0 * 1024.0));
    }

    private String formatServerDate(String value) {
        if (value == null || value.trim().isEmpty()) {
            return getString(R.string.status_unknown);
        }
        String normalized = value.trim().replace('T', ' ');
        return normalized.length() > 16 ? normalized.substring(0, 16) : normalized;
    }

    private TextView bodyText(String value) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextColor(getColor(R.color.pawprint_muted));
        view.setTextSize(13);
        view.setLineSpacing(0, 1.12f);
        return view;
    }

    private void showText(TextView view, String value) {
        view.setText(value);
        view.setVisibility(View.VISIBLE);
    }

    private void hideText(TextView view) {
        view.setVisibility(View.GONE);
        view.setText("");
    }

    private void animateViewIn(View view, int offsetDp, long startDelayMillis) {
        view.animate().cancel();
        if (!animationsEnabled()) {
            resetMotionState(view);
            return;
        }
        view.setAlpha(0f);
        view.setTranslationY(dp(offsetDp));
        view.animate()
                .alpha(1f)
                .translationY(0f)
                .setStartDelay(startDelayMillis)
                .setDuration(220L)
                .setInterpolator(new DecelerateInterpolator())
                .withLayer()
                .start();
    }

    private void resetMotionState(View view) {
        view.animate().cancel();
        view.setAlpha(1f);
        view.setTranslationY(0f);
    }

    private boolean animationsEnabled() {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.O
                || ValueAnimator.areAnimatorsEnabled();
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        outState.putInt(STATE_SELECTED_PAGE, selectedPage);
        super.onSaveInstanceState(outState);
    }

    @Override
    protected void onRestoreInstanceState(Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        int page = savedInstanceState.getInt(STATE_SELECTED_PAGE, PAGE_IDENTIFY);
        if (page < PAGE_STATUS || page > PAGE_GALLERY) {
            page = PAGE_IDENTIFY;
        }
        selectPage(page);
    }

    @Override
    protected void onDestroy() {
        clearPendingCamera(true);
        if (apiClient != null) {
            apiClient.close();
        }
        if (previewLoader != null) {
            previewLoader.close();
        }
        super.onDestroy();
    }
}
