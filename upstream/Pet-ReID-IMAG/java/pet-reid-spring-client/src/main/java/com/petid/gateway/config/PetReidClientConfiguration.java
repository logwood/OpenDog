package com.petid.gateway.config;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestClient;

@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties({PetReidProperties.class, AdminProperties.class})
public class PetReidClientConfiguration {

    @Bean
    RestClient petReidRestClient(RestClient.Builder builder, PetReidProperties properties) {
        RestClient.Builder configured = builder
                .baseUrl(properties.baseUrl().toString())
                .defaultHeader(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE);
        if (StringUtils.hasText(properties.apiKey())) {
            configured.defaultHeader("X-API-Key", properties.apiKey());
        }
        return configured.build();
    }
}
