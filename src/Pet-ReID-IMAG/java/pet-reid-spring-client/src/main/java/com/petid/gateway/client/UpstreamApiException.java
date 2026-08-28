package com.petid.gateway.client;

import org.springframework.http.HttpStatusCode;

public class UpstreamApiException extends RuntimeException {

    private final HttpStatusCode statusCode;
    private final String errorCode;
    private final String responseBody;

    public UpstreamApiException(
            HttpStatusCode statusCode,
            String errorCode,
            String message,
            String responseBody) {
        super(message);
        this.statusCode = statusCode;
        this.errorCode = errorCode;
        this.responseBody = responseBody;
    }

    public HttpStatusCode statusCode() {
        return statusCode;
    }

    public String errorCode() {
        return errorCode;
    }

    public String responseBody() {
        return responseBody;
    }
}

