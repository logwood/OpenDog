package com.petid.gateway.web;

import static org.mockito.Mockito.mock;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.converter.json.JacksonJsonHttpMessageConverter;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import com.petid.gateway.client.PetReidClient;
import com.petid.gateway.config.AdminProperties;

import tools.jackson.databind.PropertyNamingStrategies;
import tools.jackson.databind.json.JsonMapper;

class AdminControllerTest {

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        var mapper = JsonMapper.builder()
                .propertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
                .build();
        var authorizer = new AdminAuthorizer(new AdminProperties("test-secret"));
        mockMvc = MockMvcBuilders
                .standaloneSetup(new AdminController(mock(PetReidClient.class), authorizer))
                .setControllerAdvice(new ApiExceptionHandler())
                .setMessageConverters(new JacksonJsonHttpMessageConverter(mapper))
                .build();
    }

    @Test
    void rejectsMissingOrIncorrectAdminKey() throws Exception {
        mockMvc.perform(get("/v1/admin/access"))
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.error.code").value("admin_unauthorized"));

        mockMvc.perform(get("/v1/admin/access").header("X-Admin-Key", "wrong"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void acceptsConfiguredAdminKey() throws Exception {
        mockMvc.perform(get("/v1/admin/access").header("X-Admin-Key", "test-secret"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.authorized").value(true));
    }
}
