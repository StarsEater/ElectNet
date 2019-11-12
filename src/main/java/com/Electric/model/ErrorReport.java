package com.Electric.model;

public class ErrorReport {
    private String report_name;
    private String substation;
    private String type_code;
    private String tower_code;
    private String extra_time;
    private String use_time;
    private String use_search;
    private String err_time;
    private String err_search;
    private String err_how;
    private String err_name;
    private String factory;
    private String temperature;
    private String humidity;
    private String volt;
    private String err_level;
    private String device_type;
    private String err_device_type;
    private String report_id;
    private String conclusion;
    private String advice;
    private String analysis;
    private String dispose;
    private String description;
    private String situation;
    private String time_label;
    private String whole_text;

    public String getWhole_text() {
        return whole_text;
    }

    public void setWhole_text(String whole_text) {
        this.whole_text = whole_text;
    }

    private int use_time_y;
    private int use_time_m;
    private int use_time_d;
    private int err_time_y;
    private int err_time_m;
    private int err_time_d;

    public int getErr_time_d() {
        return err_time_d;
    }

    public int getErr_time_m() {
        return err_time_m;
    }

    public int getErr_time_y() {
        return err_time_y;
    }

    public int getUse_time_d() {
        return use_time_d;
    }

    public int getUse_time_m() {
        return use_time_m;
    }

    public int getUse_time_y() {
        return use_time_y;
    }

    public void setErr_time_d(int err_time_d) {
        this.err_time_d = err_time_d;
    }

    public void setErr_time_m(int err_time_m) {
        this.err_time_m = err_time_m;
    }

    public void setErr_time_y(int err_time_y) {
        this.err_time_y = err_time_y;
    }

    public void setUse_time_d(int use_time_d) {
        this.use_time_d = use_time_d;
    }

    public void setUse_time_m(int use_time_m) {
        this.use_time_m = use_time_m;
    }

    public void setUse_time_y(int use_time_y) {
        this.use_time_y = use_time_y;
    }

    public String getErr_search() {
        return err_search;
    }

    public String getUse_search() {
        return use_search;
    }

    public void setErr_search(String err_search) {
        this.err_search = err_search;
    }

    public void setUse_search(String use_search) {
        this.use_search = use_search;
    }

    public String getTime_label() {
        return time_label;
    }

    public void setTime_label(String time_label) {
        this.time_label = time_label;
    }

    public String getDescription() {

        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getDispose() {

        return dispose;
    }

    public void setDispose(String dispose) {
        this.dispose = dispose;
    }

    public String getAnalysis() {

        return analysis;
    }

    public void setAnalysis(String analysis) {
        this.analysis = analysis;
    }

    public String getConclusion() {

        return conclusion;
    }

    public void setConclusion(String conclusion) {
        this.conclusion = conclusion;
    }

    public String getAdvice() {

        return advice;
    }

    public void setAdvice(String advice) {
        this.advice = advice;
    }

    public String getErr_device_type() {

        return err_device_type;
    }

    public void setErr_device_type(String err_device_type) {
        this.err_device_type = err_device_type;
    }

    public String getFactory() {
        return factory;
    }

    public String getHumidity() {
        return humidity;
    }

    public String getTemperature() {
        return temperature;
    }

    public String getExtra_time() {
        return extra_time;
    }

    public String getReport_id() {
        return report_id;
    }

    public String getSituation() {
        return situation;
    }

    public String getDevice_type() {
        return device_type;
    }

    public String getErr_how() {
        return err_how;
    }

    public String getErr_level() {
        return err_level;
    }

    public String getErr_name() {
        return err_name;
    }

    public String getErr_time() {
        return err_time;
    }

    public String getReport_name() {
        return report_name;
    }

    public String getSubstation() {
        return substation;
    }

    public String getTower_code() {
        return tower_code;
    }

    public String getType_code() {
        return type_code;
    }

    public String getUse_time() {
        return use_time;
    }

    public String getVolt() {
        return volt;
    }

    public void setExtra_time(String extra_time) {
        this.extra_time = extra_time;
    }

    public void setFactory(String factory) {
        this.factory = factory;
    }

    public void setHumidity(String humidity) {
        this.humidity = humidity;
    }

    public void setTemperature(String temperature) {
        this.temperature = temperature;
    }

    public void setReport_id(String report_id) {
        this.report_id = report_id;
    }

    public void setSituation(String situation) {
        this.situation = situation;
    }

    public void setDevice_type(String device_type) {
        this.device_type = device_type;
    }

    public void setErr_how(String err_how) {
        this.err_how = err_how;
    }

    public void setErr_level(String err_level) {
        this.err_level = err_level;
    }

    public void setErr_name(String err_name) {
        this.err_name = err_name;
    }

    public void setErr_time(String err_time) {
        this.err_time = err_time;
    }

    public void setReport_name(String report_name) {
        this.report_name = report_name;
    }

    public void setSubstation(String substation) {
        this.substation = substation;
    }

    public void setTower_code(String tower_code) {
        this.tower_code = tower_code;
    }

    public void setType_code(String type_code) {
        this.type_code = type_code;
    }

    public void setUse_time(String use_time) {
        this.use_time = use_time;
    }

    public void setVolt(String volt) {
        this.volt = volt;
    }

}
