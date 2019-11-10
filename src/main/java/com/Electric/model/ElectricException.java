package com.Electric.model;

/**
 * Created by Muki on 2017/10/23
 */
public class ElectricException extends RuntimeException {
    private int httpStatus;
    private String userMessage;

    public ElectricException(int httpStatus, String message, String userMessage) {
        super(message);
        this.httpStatus = httpStatus;
        this.userMessage = userMessage;
    }

    public int getHttpStatus() {
        return httpStatus;
    }

    public void setHttpStatus(int httpStatus) {
        this.httpStatus = httpStatus;
    }

    public String getUserMessage() {
        return userMessage;
    }

    public void setUserMessage(String userMessage) {
        this.userMessage = userMessage;
    }
}
