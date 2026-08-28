package com.petid.gateway.client;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import tools.jackson.core.JacksonException;
import tools.jackson.databind.ObjectMapper;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.multipart.MultipartFile;

import com.petid.gateway.model.PetReidDtos.ApiErrorEnvelope;
import com.petid.gateway.model.PetReidDtos.BatchJob;
import com.petid.gateway.model.PetReidDtos.BatchListResponse;
import com.petid.gateway.model.PetReidDtos.DeleteImageResponse;
import com.petid.gateway.model.PetReidDtos.DeletePetResponse;
import com.petid.gateway.model.PetReidDtos.EnrollmentResponse;
import com.petid.gateway.model.PetReidDtos.HealthResponse;
import com.petid.gateway.model.PetReidDtos.HistoryItem;
import com.petid.gateway.model.PetReidDtos.HistoryListResponse;
import com.petid.gateway.model.PetReidDtos.IdentificationResponse;
import com.petid.gateway.model.PetReidDtos.PetDetails;
import com.petid.gateway.model.PetReidDtos.PetListResponse;
import com.petid.gateway.model.PetReidDtos.RestoreResponse;

@Component
public class HttpPetReidClient implements PetReidClient {

    private final RestClient restClient;
    private final ObjectMapper objectMapper;

    public HttpPetReidClient(RestClient petReidRestClient, ObjectMapper objectMapper) {
        this.restClient = petReidRestClient;
        this.objectMapper = objectMapper;
    }

    @Override
    public HealthResponse health() {
        return execute(() -> restClient.get()
                .uri("/health")
                .retrieve()
                .body(HealthResponse.class));
    }

    @Override
    public PetListResponse listPets() {
        return execute(() -> restClient.get()
                .uri("/v1/pets")
                .retrieve()
                .body(PetListResponse.class));
    }

    @Override
    public PetDetails getPet(String petId) {
        return execute(() -> restClient.get()
                .uri("/v1/pets/{petId}", petId)
                .retrieve()
                .body(PetDetails.class));
    }

    @Override
    public PetDetails updatePet(String petId, String displayName) {
        return execute(() -> restClient.put()
                .uri("/v1/pets/{petId}", petId)
                .contentType(MediaType.APPLICATION_JSON)
                .body(Map.of("display_name", displayName))
                .retrieve()
                .body(PetDetails.class));
    }

    @Override
    public EnrollmentResponse enroll(
            String petId,
            String displayName,
            List<MultipartFile> files) {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        if (StringUtils.hasText(displayName)) {
            body.add("display_name", displayName);
        }
        files.forEach(file -> body.add("files", new NamedByteArrayResource(file)));
        return execute(() -> restClient.post()
                .uri("/v1/pets/{petId}/images", petId)
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(EnrollmentResponse.class));
    }

    @Override
    public IdentificationResponse identify(
            MultipartFile file,
            int topK,
            Double matchThreshold,
            Double minimumMargin) {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new NamedByteArrayResource(file));
        return execute(() -> restClient.post()
                .uri(uriBuilder -> {
                    var builder = uriBuilder
                            .path("/v1/identify")
                            .queryParam("top_k", topK);
                    if (matchThreshold != null) {
                        builder.queryParam("match_threshold", matchThreshold);
                    }
                    if (minimumMargin != null) {
                        builder.queryParam("minimum_margin", minimumMargin);
                    }
                    return builder.build();
                })
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(IdentificationResponse.class));
    }

    @Override
    public ResponseEntity<byte[]> downloadImage(String petId, String imageId) {
        return execute(() -> restClient.get()
                .uri("/v1/pets/{petId}/images/{imageId}", petId, imageId)
                .retrieve()
                .toEntity(byte[].class));
    }

    @Override
    public DeleteImageResponse deleteImage(String petId, String imageId) {
        return execute(() -> restClient.delete()
                .uri("/v1/pets/{petId}/images/{imageId}", petId, imageId)
                .retrieve()
                .body(DeleteImageResponse.class));
    }

    @Override
    public DeletePetResponse deletePet(String petId) {
        return execute(() -> restClient.delete()
                .uri("/v1/pets/{petId}", petId)
                .retrieve()
                .body(DeletePetResponse.class));
    }

    @Override
    public HistoryListResponse listHistory(
            int page,
            int pageSize,
            String source,
            Boolean accepted,
            String reviewStatus,
            String petId,
            boolean hardOnly) {
        return execute(() -> restClient.get()
                .uri(uriBuilder -> {
                    var builder = uriBuilder.path("/v1/history")
                            .queryParam("page", page)
                            .queryParam("page_size", pageSize)
                            .queryParam("hard_only", hardOnly);
                    if (StringUtils.hasText(source)) builder.queryParam("source", source);
                    if (accepted != null) builder.queryParam("accepted", accepted);
                    if (StringUtils.hasText(reviewStatus)) {
                        builder.queryParam("review_status", reviewStatus);
                    }
                    if (StringUtils.hasText(petId)) builder.queryParam("pet_id", petId);
                    return builder.build();
                })
                .retrieve()
                .body(HistoryListResponse.class));
    }

    @Override
    public HistoryItem getHistory(String historyId) {
        return execute(() -> restClient.get()
                .uri("/v1/history/{historyId}", historyId)
                .retrieve()
                .body(HistoryItem.class));
    }

    @Override
    public ResponseEntity<byte[]> downloadHistoryImage(String historyId) {
        return execute(() -> restClient.get()
                .uri("/v1/history/{historyId}/image", historyId)
                .retrieve()
                .toEntity(byte[].class));
    }

    @Override
    public HistoryItem reviewHistory(String historyId, String status, String note) {
        return execute(() -> restClient.put()
                .uri("/v1/history/{historyId}/review", historyId)
                .contentType(MediaType.APPLICATION_JSON)
                .body(Map.of("status", status, "note", note == null ? "" : note))
                .retrieve()
                .body(HistoryItem.class));
    }

    @Override
    public Object deleteHistory(String historyId) {
        return execute(() -> restClient.delete()
                .uri("/v1/history/{historyId}", historyId)
                .retrieve()
                .body(Object.class));
    }

    @Override
    public BatchJob createBatch(
            String name,
            List<MultipartFile> files,
            List<String> expectedPetIds,
            int topK,
            Double matchThreshold,
            Double minimumMargin) {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("name", StringUtils.hasText(name) ? name : "批量测试");
        files.forEach(file -> body.add("files", new NamedByteArrayResource(file)));
        if (expectedPetIds != null) {
            expectedPetIds.forEach(label -> body.add("expected_pet_ids", label == null ? "" : label));
        }
        return execute(() -> restClient.post()
                .uri(uriBuilder -> {
                    var builder = uriBuilder.path("/v1/batches").queryParam("top_k", topK);
                    if (matchThreshold != null) builder.queryParam("match_threshold", matchThreshold);
                    if (minimumMargin != null) builder.queryParam("minimum_margin", minimumMargin);
                    return builder.build();
                })
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(BatchJob.class));
    }

    @Override
    public BatchListResponse listBatches(int page, int pageSize) {
        return execute(() -> restClient.get()
                .uri(uriBuilder -> uriBuilder.path("/v1/batches")
                        .queryParam("page", page)
                        .queryParam("page_size", pageSize)
                        .build())
                .retrieve()
                .body(BatchListResponse.class));
    }

    @Override
    public BatchJob getBatch(String batchId) {
        return execute(() -> restClient.get()
                .uri("/v1/batches/{batchId}", batchId)
                .retrieve()
                .body(BatchJob.class));
    }

    @Override
    public BatchJob cancelBatch(String batchId) {
        return execute(() -> restClient.delete()
                .uri("/v1/batches/{batchId}", batchId)
                .retrieve()
                .body(BatchJob.class));
    }

    @Override
    public ResponseEntity<byte[]> downloadBatchCsv(String batchId) {
        return execute(() -> restClient.get()
                .uri("/v1/batches/{batchId}/results.csv", batchId)
                .retrieve()
                .toEntity(byte[].class));
    }

    @Override
    public HistoryListResponse hardCases(int page, int pageSize, String reviewStatus) {
        return execute(() -> restClient.get()
                .uri(uriBuilder -> {
                    var builder = uriBuilder.path("/v1/hard-cases")
                            .queryParam("page", page)
                            .queryParam("page_size", pageSize);
                    if (StringUtils.hasText(reviewStatus)) {
                        builder.queryParam("review_status", reviewStatus);
                    }
                    return builder.build();
                })
                .retrieve()
                .body(HistoryListResponse.class));
    }

    @Override
    public ResponseEntity<byte[]> downloadGalleryBackup() {
        return execute(() -> restClient.get()
                .uri("/v1/gallery/backup")
                .retrieve()
                .toEntity(byte[].class));
    }

    @Override
    public RestoreResponse restoreGallery(MultipartFile file) {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new NamedByteArrayResource(file));
        return execute(() -> restClient.post()
                .uri("/v1/gallery/restore")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(RestoreResponse.class));
    }

    private <T> T execute(Supplier<T> operation) {
        try {
            T result = operation.get();
            if (result == null) {
                throw new UpstreamUnavailableException(
                        "Pet ReID service returned an empty response",
                        null);
            }
            return result;
        } catch (RestClientResponseException exception) {
            throw translate(exception);
        } catch (RestClientException exception) {
            throw new UpstreamUnavailableException(
                    "Cannot reach the Pet ReID inference service",
                    exception);
        }
    }

    private UpstreamApiException translate(RestClientResponseException exception) {
        String body = exception.getResponseBodyAsString();
        String code = "upstream_http_error";
        String message = "Pet ReID service returned HTTP " + exception.getStatusCode().value();
        try {
            ApiErrorEnvelope envelope = objectMapper.readValue(body, ApiErrorEnvelope.class);
            if (envelope != null && envelope.error() != null) {
                code = envelope.error().code();
                message = envelope.error().message();
            }
        } catch (JacksonException ignored) {
            // Keep the stable fallback while retaining the raw body on the exception.
        }
        return new UpstreamApiException(
                exception.getStatusCode(),
                code,
                message,
                body);
    }

    private static final class NamedByteArrayResource extends ByteArrayResource {

        private final String filename;

        private NamedByteArrayResource(MultipartFile file) {
            super(read(file));
            this.filename = StringUtils.hasText(file.getOriginalFilename())
                    ? file.getOriginalFilename()
                    : "upload.jpg";
        }

        @Override
        public String getFilename() {
            return filename;
        }

        private static byte[] read(MultipartFile file) {
            try {
                return file.getBytes();
            } catch (IOException exception) {
                throw new UncheckedIOException("Cannot read multipart upload", exception);
            }
        }
    }
}

