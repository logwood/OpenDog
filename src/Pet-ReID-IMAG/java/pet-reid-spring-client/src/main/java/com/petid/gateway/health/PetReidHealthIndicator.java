package com.petid.gateway.health;

import org.springframework.boot.health.contributor.Health;
import org.springframework.boot.health.contributor.HealthIndicator;
import org.springframework.stereotype.Component;

import com.petid.gateway.client.PetReidClient;

@Component("petReidInference")
public class PetReidHealthIndicator implements HealthIndicator {

    private final PetReidClient client;

    public PetReidHealthIndicator(PetReidClient client) {
        this.client = client;
    }

    @Override
    public Health health() {
        try {
            var upstream = client.health();
            return Health.up()
                    .withDetail("provider", upstream.backend().get("provider"))
                    .withDetail("modelFingerprint", upstream.modelFingerprint())
                    .withDetail("pets", upstream.gallery().pets())
                    .withDetail("referenceImages", upstream.gallery().referenceImages())
                    .build();
        } catch (RuntimeException exception) {
            return Health.down(exception).build();
        }
    }
}

