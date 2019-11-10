package com.Electric.model;

public class Node {
    private String name;
    private int value;
    private int category;

    public void setName(String name) {
        this.name = name;
    }

    public void setCategory(int category) {
        this.category = category;
    }

    public void setValue(int value) {
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public int getCategory() {
        return category;
    }

    public int getValue() {
        return value;
    }
}
