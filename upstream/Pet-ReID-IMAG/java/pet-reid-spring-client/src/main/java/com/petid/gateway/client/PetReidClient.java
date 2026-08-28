package com.petid.gateway.client;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.multipart.MultipartFile;

import com.petid.gateway.model.PetReidDtos.DeleteImageResponse;
import com.petid.gateway.model.PetReidDtos.DeletePetResponse;
import com.petid.gateway.model.PetReidDtos.EnrollmentResponse;
import com.petid.gateway.model.PetReidDtos.BatchJob;
import com.petid.gateway.model.PetReidDtos.BatchListResponse;
import com.petid.gateway.model.PetReidDtos.HealthResponse;
import com.petid.gateway.model.PetReidDtos.HistoryItem;
import com.petid.gateway.model.PetReidDtos.HistoryListResponse;
import com.petid.gateway.model.PetReidDtos.IdentificationResponse;
import com.petid.gateway.model.PetReidDtos.PetDetails;
import com.petid.gateway.model.PetReidDtos.PetListResponse;
import com.petid.gateway.model.PetReidDtos.RestoreResponse;

public interface PetReidClient {

    HealthResponse health();

    PetListResponse listPets();

    PetDetails getPet(String petId);

    PetDetails updatePet(String petId, String displayName);

    EnrollmentResponse enroll(
            String petId,
            String displayName,
            List<MultipartFile> files);

    IdentificationResponse identify(
            MultipartFile file,
            int topK,
            Double matchThreshold,
            Double minimumMargin);

    ResponseEntity<byte[]> downloadImage(String petId, String imageId);

    DeleteImageResponse deleteImage(String petId, String imageId);

    DeletePetResponse deletePet(String petId);

    HistoryListResponse listHistory(
            int page,
            int pageSize,
            String source,
            Boolean accepted,
            String reviewStatus,
            String petId,
            boolean hardOnly);

    HistoryItem getHistory(String historyId);

    ResponseEntity<byte[]> downloadHistoryImage(String historyId);

    HistoryItem reviewHistory(String historyId, String status, String note);

    Object deleteHistory(String historyId);

    BatchJob createBatch(
            String name,
            List<MultipartFile> files,
            List<String> expectedPetIds,
            int topK,
            Double matchThreshold,
            Double minimumMargin);

    BatchListResponse listBatches(int page, int pageSize);

    BatchJob getBatch(String batchId);

    BatchJob cancelBatch(String batchId);

    ResponseEntity<byte[]> downloadBatchCsv(String batchId);

    HistoryListResponse hardCases(int page, int pageSize, String reviewStatus);

    ResponseEntity<byte[]> downloadGalleryBackup();

    RestoreResponse restoreGallery(MultipartFile file);
}
