package com.Electric.controller;

import com.Electric.service.RelationService;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping(path = "/relation")
@EnableAutoConfiguration
public class RelationController {
    private final RelationService relationService;

    @Autowired
    public RelationController(RelationService relationService) {
        this.relationService = relationService;
    }

    @RequestMapping(value = "/getMap",method = RequestMethod.GET)
    public ResponseEntity<Map<String,Object>> getMap() {
        return relationService.getRelationMap();
    }

    @RequestMapping(value = "/getTable",method = RequestMethod.GET)
    public ResponseEntity<Map<String,Object>> getMap(@Param("keyword") String key) {
        return relationService.getRelationTable(key);
    }

}
