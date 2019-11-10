package com.Electric.model;/**
 * Created by Jfcui on 2018/11/30
 */

public class ErrorReportPics {
    String[] pics;

    public ErrorReportPics(String pics) {
        this.pics = pics.split(",");
    }

    public String[] getPics() {
        return pics;
    }

    public void setPics(String pics) {
        this.pics = pics.split(",");
    }
}
