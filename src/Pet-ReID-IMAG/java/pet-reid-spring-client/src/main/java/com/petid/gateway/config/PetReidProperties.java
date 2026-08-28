package com.petid.gateway.config;

import java.net.URI;
import java.util.Objects;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "pet-reid")
public record PetReidProperties(URI baseUrl, String apiKey) {

    public PetReidProperties {
        Objects.requireNonNull(baseUrl, "pet-reid.base-url is required");
        apiKey = apiKey == null ? "" : apiKey.trim();
    }
}

