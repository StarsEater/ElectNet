package com.Electric.model;

public class Link {
    private String source;
    private String target;
    private Double support;
    private Double confidence;

    public String getSource() {
        return source;
    }

    public String getTarget() {
        return target;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public void setTarget(String target) {
        this.target = target;
    }

    public Double getConfidence() {
        return confidence;
    }

    public Double getSupport() {
        return support;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public void setSupport(Double support) {
        this.support = support;
    }
}
