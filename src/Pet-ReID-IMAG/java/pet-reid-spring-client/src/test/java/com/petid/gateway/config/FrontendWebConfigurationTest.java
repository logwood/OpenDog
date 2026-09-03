package com.petid.gateway.config;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpHeaders;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;

class FrontendWebConfigurationTest {

    @Test
    void corsIsDisabledByDefault() {
        var registry = new InspectableCorsRegistry();

        new FrontendWebConfiguration(new FrontendCorsProperties(List.of()))
                .addCorsMappings(registry);

        assertThat(registry.configurations()).isEmpty();
    }

    @Test
    void registersExactNormalizedFrontendOrigins() {
        var properties = new FrontendCorsProperties(List.of(
                " HTTP://LOCALHOST:5173/ ",
                "http://localhost:5173",
                "https://app.example"));
        var registry = new InspectableCorsRegistry();

        new FrontendWebConfiguration(properties).addCorsMappings(registry);

        assertThat(properties.allowedOrigins())
                .containsExactly("http://localhost:5173", "https://app.example");
        CorsConfiguration cors = registry.configurations().get("/v1/**");
        assertThat(cors).isNotNull();
        assertThat(cors.getAllowedOrigins())
                .containsExactly("http://localhost:5173", "https://app.example");
        assertThat(cors.getAllowedMethods())
                .containsExactly("GET", "POST", "PATCH", "DELETE", "OPTIONS");
        assertThat(cors.getAllowedHeaders())
                .containsExactly(
                        HttpHeaders.ACCEPT,
                        HttpHeaders.CONTENT_TYPE,
                        "X-Admin-Key");
        assertThat(cors.getExposedHeaders()).containsExactly(HttpHeaders.CONTENT_DISPOSITION);
        assertThat(cors.getAllowCredentials()).isFalse();
        assertThat(cors.getMaxAge()).isEqualTo(3600);
    }

    @Test
    void rejectsUrlsThatAreNotBrowserOrigins() {
        assertThatThrownBy(() -> new FrontendCorsProperties(
                List.of("https://app.example/not-an-origin")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("HTTP(S) origins");
    }

    private static final class InspectableCorsRegistry extends CorsRegistry {

        Map<String, CorsConfiguration> configurations() {
            return getCorsConfigurations();
        }
    }
}
