package com.petid.gateway.client;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.hamcrest.Matchers.containsString;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.queryParam;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.converter.json.JacksonJsonHttpMessageConverter;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import tools.jackson.databind.PropertyNamingStrategies;
import tools.jackson.databind.json.JsonMapper;

class HttpPetReidClientTest {

    private MockRestServiceServer server;
    private HttpPetReidClient client;

    @BeforeEach
    void setUp() {
        JsonMapper mapper = JsonMapper.builder()
                .propertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
                .build();
        RestClient.Builder builder = RestClient.builder()
                .baseUrl("http://pet-reid.test")
                .configureMessageConverters(converters -> converters
                        .withJsonConverter(new JacksonJsonHttpMessageConverter(mapper)));
        server = MockRestServiceServer.bindTo(builder).bufferContent().build();
        client = new HttpPetReidClient(builder.build(), mapper);
    }

    @Test
    void readsSnakeCaseHealthResponse() {
        server.expect(requestTo("http://pet-reid.test/health"))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess("""
                        {
                          "status": "ok",
                          "model_fingerprint": "abc123",
                          "backend": {"provider": "CUDAExecutionProvider"},
                          "gallery": {"pets": 2, "reference_images": 4}
                        }
                        """, MediaType.APPLICATION_JSON));

        var response = client.health();

        assertThat(response.status()).isEqualTo("ok");
        assertThat(response.modelFingerprint()).isEqualTo("abc123");
        assertThat(response.backend()).containsEntry("provider", "CUDAExecutionProvider");
        assertThat(response.gallery().pets()).isEqualTo(2);
        assertThat(response.gallery().referenceImages()).isEqualTo(4);
        server.verify();
    }

    @Test
    void forwardsEnrollmentAsMultipart() {
        var upload = new MockMultipartFile(
                "files",
                "buddy.jpg",
                MediaType.IMAGE_JPEG_VALUE,
                new byte[] {1, 2, 3, 4});
        server.expect(requestTo("http://pet-reid.test/v1/pets/dog-1/images"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(content().string(containsString("name=\"display_name\"")))
                .andExpect(content().string(containsString("Buddy")))
                .andExpect(content().string(containsString("filename=\"buddy.jpg\"")))
                .andRespond(withSuccess("""
                        {
                          "pet": {
                            "pet_id": "dog-1",
                            "display_name": "Buddy",
                            "created_at": "2026-08-26T12:00:00Z",
                            "updated_at": "2026-08-26T12:00:00Z",
                            "reference_count": 1,
                            "images": []
                          },
                          "added_image_ids": ["img-1"],
                          "duplicate_image_ids": []
                        }
                        """, MediaType.APPLICATION_JSON));

        var response = client.enroll("dog-1", "Buddy", List.of(upload));

        assertThat(response.pet().petId()).isEqualTo("dog-1");
        assertThat(response.pet().referenceCount()).isEqualTo(1);
        assertThat(response.addedImageIds()).containsExactly("img-1");
        server.verify();
    }

    @Test
    void forwardsIdentificationOptions() {
        var upload = new MockMultipartFile(
                "file",
                "query.jpg",
                MediaType.IMAGE_JPEG_VALUE,
                new byte[] {5, 6, 7});
        server.expect(requestTo(containsString("http://pet-reid.test/v1/identify")))
                .andExpect(method(HttpMethod.POST))
                .andExpect(queryParam("top_k", "3"))
                .andExpect(queryParam("match_threshold", "0.7"))
                .andExpect(queryParam("minimum_margin", "0.05"))
                .andExpect(content().string(containsString("filename=\"query.jpg\"")))
                .andRespond(withSuccess("""
                        {
                          "decision": "matched",
                          "accepted": true,
                          "predicted_pet_id": "dog-1",
                          "predicted_display_name": "Buddy",
                          "top1_score": 0.91,
                          "margin": 0.22,
                          "match_threshold": 0.7,
                          "minimum_margin": 0.05,
                          "candidates": [{
                            "pet_id": "dog-1",
                            "display_name": "Buddy",
                            "score": 0.91,
                            "reference_count": 2
                          }],
                          "query": {"width": 800, "height": 200}
                        }
                        """, MediaType.APPLICATION_JSON));

        var response = client.identify(upload, 3, 0.7, 0.05);

        assertThat(response.accepted()).isTrue();
        assertThat(response.predictedPetId()).isEqualTo("dog-1");
        assertThat(response.top1Score()).isEqualTo(0.91);
        assertThat(response.query()).containsEntry("width", 800);
        server.verify();
    }

    @Test
    void forwardsHistoryReviewThroughCompatibleUpstreamMethod() {
        server.expect(requestTo("http://pet-reid.test/v1/history/history-1/review"))
                .andExpect(method(HttpMethod.PUT))
                .andExpect(content().json("""
                        {"status":"incorrect","note":"wrong match"}
                        """))
                .andRespond(withSuccess("""
                        {
                          "history_id": "history-1",
                          "created_at": "2026-08-28T00:00:00Z",
                          "source": "single",
                          "status": "succeeded",
                          "filename": "query.jpg",
                          "sha256": "abc",
                          "byte_size": 3,
                          "image_available": true,
                          "accepted": true,
                          "predicted_pet_id": "dog-1",
                          "model_fingerprint": "model-1",
                          "gallery_snapshot": {"pets": 2, "reference_images": 4},
                          "review_status": "incorrect",
                          "review_note": "wrong match",
                          "hard_case_reasons": []
                        }
                        """, MediaType.APPLICATION_JSON));

        var response = client.reviewHistory("history-1", "incorrect", "wrong match");

        assertThat(response.reviewStatus()).isEqualTo("incorrect");
        assertThat(response.reviewNote()).isEqualTo("wrong match");
        server.verify();
    }

    @Test
    void preservesStructuredUpstreamErrors() {
        server.expect(requestTo("http://pet-reid.test/v1/pets/missing"))
                .andRespond(withStatus(HttpStatus.NOT_FOUND)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("""
                                {
                                  "error": {
                                    "code": "pet_not_found",
                                    "message": "Unknown pet: missing",
                                    "details": {"pet_id": "missing"}
                                  }
                                }
                                """));

        assertThatThrownBy(() -> client.getPet("missing"))
                .isInstanceOfSatisfying(UpstreamApiException.class, exception -> {
                    assertThat(exception.statusCode().value()).isEqualTo(404);
                    assertThat(exception.errorCode()).isEqualTo("pet_not_found");
                    assertThat(exception.getMessage()).isEqualTo("Unknown pet: missing");
                });
        server.verify();
    }
}
