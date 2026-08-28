package com.petid.gateway;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest(properties = "pet-reid.base-url=http://127.0.0.1:9")
class PetReidApplicationTest {

    @Test
    void contextLoads() {
    }
}
