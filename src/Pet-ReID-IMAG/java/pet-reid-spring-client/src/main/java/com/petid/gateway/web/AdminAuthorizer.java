package com.petid.gateway.web;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import com.petid.gateway.config.AdminProperties;
import com.petid.gateway.model.PetReidDtos.ApiError;
import com.petid.gateway.model.PetReidDtos.ApiErrorEnvelope;

@Component
public class AdminAuthorizer {

    private final String apiKey;

    public AdminAuthorizer(AdminProperties properties) {
        this.apiKey = properties.apiKey();
    }

    public boolean configured() {
        return StringUtils.hasText(apiKey);
    }

    public void require(String provided) {
        if (!configured() || !StringUtils.hasText(provided)
                || !MessageDigest.isEqual(
                        apiKey.getBytes(StandardCharsets.UTF_8),
                        provided.getBytes(StandardCharsets.UTF_8))) {
            throw new AdminAuthorizationException();
        }
    }

    public static final class AdminAuthorizationException extends RuntimeException {

        private static final long serialVersionUID = 1L;

        public AdminAuthorizationException() {
            super("A valid X-Admin-Key header is required");
        }

        public ApiErrorEnvelope response() {
            return new ApiErrorEnvelope(new ApiError(
                    "admin_unauthorized",
                    getMessage(),
                    java.util.Map.of()));
        }

        public HttpStatus status() {
            return HttpStatus.UNAUTHORIZED;
        }
    }
}
