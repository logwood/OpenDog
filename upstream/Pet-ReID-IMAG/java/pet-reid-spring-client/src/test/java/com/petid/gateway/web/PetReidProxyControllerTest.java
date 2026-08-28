package com.petid.gateway.web;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.converter.json.JacksonJsonHttpMessageConverter;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.multipart.MultipartFile;

import com.petid.gateway.client.PetReidClient;
import com.petid.gateway.client.UpstreamApiException;
import com.petid.gateway.model.PetReidDtos.EnrollmentResponse;
import com.petid.gateway.model.PetReidDtos.PetDetails;

import tools.jackson.databind.PropertyNamingStrategies;
import tools.jackson.databind.json.JsonMapper;

class PetReidProxyControllerTest {

    private PetReidClient client;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        client = mock(PetReidClient.class);
        JsonMapper mapper = JsonMapper.builder()
                .propertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
                .build();
        mockMvc = MockMvcBuilders
                .standaloneSetup(new PetReidProxyController(client))
                .setControllerAdvice(new ApiExceptionHandler())
                .setMessageConverters(new JacksonJsonHttpMessageConverter(mapper))
                .build();
    }

    @Test
    void exposesImageEnrollmentApi() throws Exception {
        var pet = new PetDetails(
                "dog-1",
                "Buddy",
                "2026-08-26T12:00:00Z",
                "2026-08-26T12:00:00Z",
                1,
                List.of());
        when(client.enroll(eq("dog-1"), eq("Buddy"), anyList()))
                .thenReturn(new EnrollmentResponse(pet, List.of("img-1"), List.of()));
        var upload = new MockMultipartFile(
                "files",
                "buddy.jpg",
                MediaType.IMAGE_JPEG_VALUE,
                new byte[] {1, 2, 3});

        mockMvc.perform(multipart("/v1/pets/dog-1/images")
                        .file(upload)
                        .param("display_name", "Buddy"))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.pet.pet_id").value("dog-1"))
                .andExpect(jsonPath("$.pet.reference_count").value(1))
                .andExpect(jsonPath("$.added_image_ids[0]").value("img-1"));

        verify(client).enroll(eq("dog-1"), eq("Buddy"), anyList());
    }

    @Test
    void returnsStableErrorEnvelopeForUpstreamFailure() throws Exception {
        when(client.identify(any(MultipartFile.class), eq(3), isNull(), isNull()))
                .thenThrow(new UpstreamApiException(
                        HttpStatus.CONFLICT,
                        "gallery_empty",
                        "No enrolled pets",
                        "{}"));
        var upload = new MockMultipartFile(
                "file",
                "query.jpg",
                MediaType.IMAGE_JPEG_VALUE,
                new byte[] {4, 5, 6});

        mockMvc.perform(multipart("/v1/identify")
                        .file(upload)
                        .param("top_k", "3"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.error.code").value("gallery_empty"))
                .andExpect(jsonPath("$.error.message").value("No enrolled pets"));
    }
}
