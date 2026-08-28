package com.petid.gateway.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "admin")
public record AdminProperties(String apiKey) {

    public AdminProperties {
        apiKey = apiKey == null ? "" : apiKey.trim();
    }
}
