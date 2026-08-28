package com.petid.gateway.model;

import java.util.List;
import java.util.Map;

public final class PetReidDtos {

    private PetReidDtos() {
    }

    public record PetImage(
            String imageId,
            String originalFilename,
            String contentType,
            int width,
            int height,
            long byteSize,
            String sha256,
            Map<String, Object> quality,
            String createdAt) {
    }

    public record PetSummary(
            String petId,
            String displayName,
            String createdAt,
            String updatedAt,
            int referenceCount) {
    }

    public record PetDetails(
            String petId,
            String displayName,
            String createdAt,
            String updatedAt,
            int referenceCount,
            List<PetImage> images) {
    }

    public record PetListResponse(List<PetSummary> pets, int count) {
    }

    public record EnrollmentResponse(
            PetDetails pet,
            List<String> addedImageIds,
            List<String> duplicateImageIds) {
    }

    public record Candidate(
            String petId,
            String displayName,
            double score,
            int referenceCount) {
    }

    public record IdentificationResponse(
            String decision,
            boolean accepted,
            String predictedPetId,
            String predictedDisplayName,
            double top1Score,
            Double margin,
            Double matchThreshold,
            double minimumMargin,
            List<Candidate> candidates,
            Map<String, Object> query,
            Double latencyMs,
            String modelFingerprint,
            GalleryHealth gallerySnapshot,
            Map<String, Object> diagnostics,
            List<String> hardCaseReasons,
            String historyId) {
    }

    public record GalleryHealth(int pets, int referenceImages) {
    }

    public record HealthResponse(
            String status,
            String modelFingerprint,
            Map<String, Object> backend,
            GalleryHealth gallery,
            Map<String, Object> operations) {
    }

    public record DeleteImageResponse(
            String petId,
            String deletedImageId,
            int remainingReferences,
            boolean petDeleted) {
    }

    public record DeletePetResponse(String deletedPetId, int deletedImages) {
    }

    public record HistoryItem(
            String historyId,
            String createdAt,
            String source,
            String batchId,
            String status,
            String filename,
            String sha256,
            Integer width,
            Integer height,
            long byteSize,
            boolean imageAvailable,
            String expectedPetId,
            Boolean accepted,
            String predictedPetId,
            String predictedDisplayName,
            Double top1Score,
            Double margin,
            Double matchThreshold,
            Double minimumMargin,
            Double latencyMs,
            String modelFingerprint,
            GalleryHealth gallerySnapshot,
            String reviewStatus,
            String reviewNote,
            String reviewedAt,
            List<String> hardCaseReasons,
            Map<String, Object> error,
            Map<String, Object> result) {
    }

    public record HistoryListResponse(
            List<HistoryItem> items,
            long total,
            int page,
            int pageSize) {
    }

    public record HistoryReviewRequest(
            @jakarta.validation.constraints.Pattern(
                    regexp = "^(unreviewed|correct|incorrect|uncertain)$") String status,
            @jakarta.validation.constraints.Size(max = 1000) String note) {
    }

    public record PetUpdateRequest(
            @jakarta.validation.constraints.NotBlank
            @jakarta.validation.constraints.Size(max = 128) String displayName) {
    }

    public record BatchJob(
            String batchId,
            String name,
            String status,
            String createdAt,
            String startedAt,
            String finishedAt,
            int total,
            int completed,
            int succeeded,
            int failed,
            boolean cancelRequested,
            String modelFingerprint,
            Map<String, Object> parameters,
            Map<String, Object> metrics,
            String errorMessage,
            List<HistoryItem> results) {
    }

    public record BatchListResponse(
            List<BatchJob> items,
            long total,
            int page,
            int pageSize) {
    }

    public record RestoreResponse(
            int pets,
            int addedImages,
            int duplicateImages,
            String mode) {
    }

    public record ApiError(String code, String message, Map<String, Object> details) {
    }

    public record ApiErrorEnvelope(ApiError error) {
    }
}
