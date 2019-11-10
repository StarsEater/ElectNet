package com.Electric.model;

public class Decision {
    String kind;
    String param;
    String description;
    String analyse;
    String advice;
    String report;

    public String getAdvice() {
        return advice;
    }

    public String getAnalyse() {
        return analyse;
    }

    public String getDescription() {
        return description;
    }

    public String getKind() {
        return kind;
    }

    public String getParam() {
        return param;
    }

    public String getReport() {
        return report;
    }

    public void setAdvice(String advice) {
        this.advice = advice;
    }

    public void setAnalyse(String analyse) {
        this.analyse = analyse;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public void setKind(String kind) {
        this.kind = kind;
    }

    public void setParam(String param) {
        this.param = param;
    }

    public void setReport(String report) {
        this.report = report;
    }
}
