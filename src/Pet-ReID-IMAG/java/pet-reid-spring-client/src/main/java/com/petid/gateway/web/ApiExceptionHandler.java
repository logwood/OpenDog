package com.petid.gateway.web;

import java.io.UncheckedIOException;
import java.util.Map;

import jakarta.validation.ConstraintViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import com.petid.gateway.client.UpstreamApiException;
import com.petid.gateway.client.UpstreamUnavailableException;
import com.petid.gateway.model.PetReidDtos.ApiError;
import com.petid.gateway.model.PetReidDtos.ApiErrorEnvelope;
import com.petid.gateway.web.AdminAuthorizer.AdminAuthorizationException;

@RestControllerAdvice
public class ApiExceptionHandler {

    @ExceptionHandler(AdminAuthorizationException.class)
    public ResponseEntity<ApiErrorEnvelope> adminUnauthorized(
            AdminAuthorizationException exception) {
        return ResponseEntity.status(exception.status()).body(exception.response());
    }

    @ExceptionHandler(UpstreamApiException.class)
    public ResponseEntity<ApiErrorEnvelope> upstreamError(UpstreamApiException exception) {
        return ResponseEntity
                .status(exception.statusCode())
                .body(error(exception.errorCode(), exception.getMessage(), Map.of()));
    }

    @ExceptionHandler(UpstreamUnavailableException.class)
    public ResponseEntity<ApiErrorEnvelope> unavailable(UpstreamUnavailableException exception) {
        return ResponseEntity
                .status(HttpStatus.BAD_GATEWAY)
                .body(error("upstream_unavailable", exception.getMessage(), Map.of()));
    }

    @ExceptionHandler({
            ConstraintViolationException.class,
            MethodArgumentNotValidException.class,
            IllegalArgumentException.class,
            UncheckedIOException.class
    })
    public ResponseEntity<ApiErrorEnvelope> invalidRequest(Exception exception) {
        return ResponseEntity
                .badRequest()
                .body(error("invalid_request", exception.getMessage(), Map.of()));
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ApiErrorEnvelope> uploadTooLarge(
            MaxUploadSizeExceededException exception) {
        return ResponseEntity
                .status(HttpStatus.PAYLOAD_TOO_LARGE)
                .body(error("upload_too_large", exception.getMessage(), Map.of()));
    }

    private static ApiErrorEnvelope error(
            String code,
            String message,
            Map<String, Object> details) {
        return new ApiErrorEnvelope(new ApiError(code, message, details));
    }
}
