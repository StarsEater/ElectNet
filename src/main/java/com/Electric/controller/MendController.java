package com.Electric.controller;


import com.Electric.model.ErrorReport;
import com.Electric.service.MendService;
import com.Electric.service.ReportService;
import com.alibaba.fastjson.JSONObject;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;

@Controller
@RequestMapping(path = "/report")
@EnableAutoConfiguration
public class MendController {
    private final MendService mendService;
    private final String baseurl = "http://localhost:443";

    @Autowired
    public MendController(MendService mendService) {
        this.mendService = mendService;
    }

    @RequestMapping(value = "/mend", method = RequestMethod.GET)
    public String getAdvice(Model model) {
        model.addAttribute("baseurl", baseurl);
        return "mend";
    }
    @ResponseBody
    @PostMapping("/ai")
    public ResponseEntity<Map<String,Object>> getAdviceAi(
            @RequestBody JSONObject jsonObject) throws IOException, InterruptedException {
        System.out.println("xswl"+jsonObject);
        return mendService.getAiAdvice(jsonObject.get("desc").toString());
    }
}
