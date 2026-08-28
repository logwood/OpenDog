package com.petid.gateway.web;

import java.util.List;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.petid.gateway.client.PetReidClient;
import com.petid.gateway.model.PetReidDtos.DeleteImageResponse;
import com.petid.gateway.model.PetReidDtos.DeletePetResponse;
import com.petid.gateway.model.PetReidDtos.EnrollmentResponse;
import com.petid.gateway.model.PetReidDtos.HealthResponse;
import com.petid.gateway.model.PetReidDtos.HistoryItem;
import com.petid.gateway.model.PetReidDtos.HistoryListResponse;
import com.petid.gateway.model.PetReidDtos.HistoryReviewRequest;
import com.petid.gateway.model.PetReidDtos.IdentificationResponse;
import com.petid.gateway.model.PetReidDtos.PetDetails;
import com.petid.gateway.model.PetReidDtos.PetListResponse;
import com.petid.gateway.model.PetReidDtos.PetUpdateRequest;

@Validated
@RestController
@RequestMapping("/v1")
public class PetReidProxyController {

    private static final String PET_ID = "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$";

    private final PetReidClient client;

    public PetReidProxyController(PetReidClient client) {
        this.client = client;
    }

    @GetMapping("/upstream-health")
    public HealthResponse upstreamHealth() {
        return client.health();
    }

    @GetMapping("/pets")
    public PetListResponse listPets() {
        return client.listPets();
    }

    @GetMapping("/pets/{petId}")
    public PetDetails getPet(@PathVariable @Pattern(regexp = PET_ID) String petId) {
        return client.getPet(petId);
    }

    @PatchMapping("/pets/{petId}")
    public PetDetails updatePet(
            @PathVariable @Pattern(regexp = PET_ID) String petId,
            @jakarta.validation.Valid
            @org.springframework.web.bind.annotation.RequestBody PetUpdateRequest request) {
        return client.updatePet(petId, request.displayName());
    }

    @PostMapping(
            path = "/pets/{petId}/images",
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<EnrollmentResponse> enroll(
            @PathVariable @Pattern(regexp = PET_ID) String petId,
            @RequestPart("files") @Size(min = 1, max = 8) List<MultipartFile> files,
            @RequestParam(name = "display_name", required = false)
            @Size(max = 128) String displayName) {
        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(client.enroll(petId, displayName, files));
    }

    @PostMapping(path = "/identify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public IdentificationResponse identify(
            @RequestPart("file") MultipartFile file,
            @RequestParam(name = "top_k", defaultValue = "5") @Min(1) @Max(50) int topK,
            @RequestParam(name = "match_threshold", required = false)
            @DecimalMin("-1.0") @DecimalMax("1.0") Double matchThreshold,
            @RequestParam(name = "minimum_margin", required = false)
            @DecimalMin("0.0") @DecimalMax("2.0") Double minimumMargin) {
        return client.identify(file, topK, matchThreshold, minimumMargin);
    }

    @GetMapping("/pets/{petId}/images/{imageId}")
    public ResponseEntity<byte[]> downloadImage(
            @PathVariable @Pattern(regexp = PET_ID) String petId,
            @PathVariable String imageId) {
        return client.downloadImage(petId, imageId);
    }

    @DeleteMapping("/pets/{petId}/images/{imageId}")
    public DeleteImageResponse deleteImage(
            @PathVariable @Pattern(regexp = PET_ID) String petId,
            @PathVariable String imageId) {
        return client.deleteImage(petId, imageId);
    }

    @DeleteMapping("/pets/{petId}")
    public DeletePetResponse deletePet(
            @PathVariable @Pattern(regexp = PET_ID) String petId) {
        return client.deletePet(petId);
    }

    @GetMapping("/history")
    public HistoryListResponse listHistory(
            @RequestParam(name = "page", defaultValue = "1") @Min(1) int page,
            @RequestParam(name = "page_size", defaultValue = "25") @Min(1) @Max(200) int pageSize,
            @RequestParam(name = "source", required = false) String source,
            @RequestParam(name = "accepted", required = false) Boolean accepted,
            @RequestParam(name = "review_status", required = false) String reviewStatus,
            @RequestParam(name = "pet_id", required = false) String petId) {
        return client.listHistory(page, pageSize, source, accepted, reviewStatus, petId, false);
    }

    @GetMapping("/history/{historyId}")
    public HistoryItem getHistory(@PathVariable String historyId) {
        return client.getHistory(historyId);
    }

    @GetMapping("/history/{historyId}/image")
    public ResponseEntity<byte[]> downloadHistoryImage(@PathVariable String historyId) {
        return client.downloadHistoryImage(historyId);
    }

    @PatchMapping("/history/{historyId}/review")
    public HistoryItem reviewHistory(
            @PathVariable String historyId,
            @jakarta.validation.Valid
            @org.springframework.web.bind.annotation.RequestBody HistoryReviewRequest request) {
        return client.reviewHistory(historyId, request.status(), request.note());
    }

    @DeleteMapping("/history/{historyId}")
    public Object deleteHistory(@PathVariable String historyId) {
        return client.deleteHistory(historyId);
    }
}
