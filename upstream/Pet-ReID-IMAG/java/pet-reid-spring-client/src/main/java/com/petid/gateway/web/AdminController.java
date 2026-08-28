package com.petid.gateway.web;

import java.util.List;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Size;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.multipart.MultipartFile;

import com.petid.gateway.client.PetReidClient;
import com.petid.gateway.model.PetReidDtos.BatchJob;
import com.petid.gateway.model.PetReidDtos.BatchListResponse;
import com.petid.gateway.model.PetReidDtos.HistoryListResponse;
import com.petid.gateway.model.PetReidDtos.RestoreResponse;

@Validated
@RestController
@RequestMapping("/v1/admin")
public class AdminController {

    private final PetReidClient client;
    private final AdminAuthorizer authorizer;

    public AdminController(PetReidClient client, AdminAuthorizer authorizer) {
        this.client = client;
        this.authorizer = authorizer;
    }

    @GetMapping("/access")
    public java.util.Map<String, Boolean> access(
            @RequestHeader(name = "X-Admin-Key", required = false) String adminKey) {
        authorizer.require(adminKey);
        return java.util.Map.of("authorized", true);
    }

    @PostMapping(path = "/batches", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<BatchJob> createBatch(
            @RequestHeader(name = "X-Admin-Key", required = false) String adminKey,
            @RequestPart("files") @Size(min = 1, max = 1000) List<MultipartFile> files,
            @RequestParam(name = "name", defaultValue = "批量测试") @Size(max = 128) String name,
            @RequestParam(name = "expected_pet_ids", required = false) List<String> expectedPetIds,
            @RequestParam(name = "top_k", defaultValue = "5") @Min(1) @Max(50) int topK,
            @RequestParam(name = "match_threshold", required = false)
            @DecimalMin("-1.0") @DecimalMax("1.0") Double matchThreshold,
            @RequestParam(name = "minimum_margin", required = false)
            @DecimalMin("0.0") @DecimalMax("2.0") Double minimumMargin) {
        authorizer.require(adminKey);
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(client.createBatch(
                name, files, expectedPetIds, topK, matchThreshold, minimumMargin));
    }

    @GetMapping("/batches")
    public BatchListResponse listBatches(
            @RequestHeader(name = "X-Admin-Key", required = false) String adminKey,
            @RequestParam(name = "page", defaultValue = "1") @Min(1) int page,
            @RequestParam(name = "page_size", defaultValue = "20") @Min(1) @Max(100) int pageSize) {
        authorizer.require(adminKey);
        return client.listBatches(page, pageSize);
    }

    @GetMapping("/batches/{batchId}")
    public BatchJob getBatch(
            @RequestHeader(name = "X-Admin-Key", required = false) String adminKey,
            @PathVariable String batchId) {
        authorizer.require(adminKey);
        return client.getBatch(batchId);
    }

    @DeleteMapping("/batches/{batchId}")
    public BatchJob cancelBatch(
            @RequestHeader(name = "X-Admin-Key", required = false) String adminKey,
            @PathVariable String batchId) {
        authorizer.require(adminKey);
        return client.cancelBatch(batchId);
    }

    @GetMapping("/batches/{batchId}/results.csv")
    public ResponseEntity<byte[]> downloadBatchCsv(
            @RequestHeader(name = "X-Admin-Key", required = false) String adminKey,
            @PathVariable String batchId) {
        authorizer.require(adminKey);
        return client.downloadBatchCsv(batchId);
    }

    @GetMapping("/hard-cases")
    public HistoryListResponse hardCases(
            @RequestHeader(name = "X-Admin-Key", required = false) String adminKey,
            @RequestParam(name = "page", defaultValue = "1") @Min(1) int page,
            @RequestParam(name = "page_size", defaultValue = "50") @Min(1) @Max(200) int pageSize,
            @RequestParam(name = "review_status", required = false) String reviewStatus) {
        authorizer.require(adminKey);
        return client.hardCases(page, pageSize, reviewStatus);
    }

    @GetMapping("/gallery/backup")
    public ResponseEntity<byte[]> backupGallery(
            @RequestHeader(name = "X-Admin-Key", required = false) String adminKey) {
        authorizer.require(adminKey);
        return client.downloadGalleryBackup();
    }

    @PostMapping(path = "/gallery/restore", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public RestoreResponse restoreGallery(
            @RequestHeader(name = "X-Admin-Key", required = false) String adminKey,
            @RequestPart("file") MultipartFile file) {
        authorizer.require(adminKey);
        return client.restoreGallery(file);
    }
}
