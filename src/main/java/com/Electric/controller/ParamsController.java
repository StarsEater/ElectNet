package com.Electric.controller;


import com.Electric.service.ParamsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping(path = "/params")
@EnableAutoConfiguration
public class ParamsController {
    private final ParamsService paramsService;

    @Autowired
    public ParamsController(ParamsService paramsService) {
        this.paramsService = paramsService;
    }
    // 决策检索
    @RequestMapping(value = "/getParams", method = RequestMethod.GET)
    public ResponseEntity<Map<String, Object>> getParams() {
        return paramsService.getSelectParams();
    }

    @RequestMapping(value = "/getDecisions", method = RequestMethod.GET)
    public ResponseEntity<Map<String, Object>> getDecisions(@RequestParam("kind") String kind, @RequestParam("param") String param, @RequestParam(value = "desc", required = false) String desc) {
        return paramsService.getDecisions(kind, param, desc);
    }
}
