package com.petid.gateway.config;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "frontend.cors")
public record FrontendCorsProperties(List<String> allowedOrigins) {

    public FrontendCorsProperties {
        allowedOrigins = allowedOrigins == null
                ? List.of()
                : allowedOrigins.stream()
                        .filter(Objects::nonNull)
                        .map(String::trim)
                        .filter(origin -> !origin.isEmpty())
                        .map(FrontendCorsProperties::normalizeOrigin)
                        .distinct()
                        .toList();
    }

    private static String normalizeOrigin(String origin) {
        URI parsed;
        try {
            parsed = new URI(origin);
        } catch (URISyntaxException exception) {
            throw new IllegalArgumentException(
                    "frontend.cors.allowed-origins contains an invalid URI: " + origin,
                    exception);
        }
        String scheme = parsed.getScheme();
        String path = parsed.getRawPath();
        boolean http = scheme != null
                && ("http".equalsIgnoreCase(scheme) || "https".equalsIgnoreCase(scheme));
        boolean originOnly = (path == null || path.isEmpty() || "/".equals(path))
                && parsed.getRawQuery() == null
                && parsed.getRawFragment() == null
                && parsed.getRawUserInfo() == null;
        if (!http || parsed.getHost() == null || !originOnly) {
            throw new IllegalArgumentException(
                    "frontend.cors.allowed-origins must contain only HTTP(S) origins: "
                            + origin);
        }
        try {
            return new URI(
                    scheme.toLowerCase(Locale.ROOT),
                    null,
                    parsed.getHost().toLowerCase(Locale.ROOT),
                    parsed.getPort(),
                    null,
                    null,
                    null)
                    .toASCIIString();
        } catch (URISyntaxException exception) {
            throw new IllegalArgumentException("Cannot normalize frontend origin: " + origin,
                    exception);
        }
    }
}
