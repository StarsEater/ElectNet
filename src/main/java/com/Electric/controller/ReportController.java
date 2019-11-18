package com.Electric.controller;


import com.Electric.model.ErrorReport;
import com.Electric.model.ErrorReportPics;
import com.Electric.service.ReportService;

import com.alibaba.fastjson.JSONObject;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * Created by shuwan on 2018/11/18
 */

@Controller
@RequestMapping(path = "/report")
@EnableAutoConfiguration
public class ReportController {
    private final ReportService reportService;
    private final String baseurl = "http://localhost:443";

    @Autowired
    public ReportController(ReportService reportService) {
        this.reportService = reportService;
    }

    @RequestMapping(value = "/getUserByGet",method = RequestMethod.GET)
    public String getUserByGet(@RequestParam(value = "userName") String userName) {
        return "Hello "+userName;
    }

//    private String getClientIP(HttpServletRequest request) {
//        String ip = request.getHeader("X-Forwarded-For");
//
//        if (ip == null || ip.length() == 0 || "unknown".equalsIgnoreCase(ip)) {
//            if (ip == null || ip.length() == 0 || "unknown".equalsIgnoreCase(ip)) {
//                ip = request.getHeader("Proxy-Client-IP");
//            }
//            if (ip == null || ip.length() == 0 || "unknown".equalsIgnoreCase(ip)) {
//                ip = request.getHeader("WL-Proxy-Client-IP");
//            }
//            if (ip == null || ip.length() == 0 || "unknown".equalsIgnoreCase(ip)) {
//                ip = request.getHeader("HTTP_CLIENT_IP");
//            }
//            if (ip == null || ip.length() == 0 || "unknown".equalsIgnoreCase(ip)) {
//                ip = request.getHeader("HTTP_X_FORWARDED_FOR");
//            }
//            if (ip == null || ip.length() == 0 || "unknown".equalsIgnoreCase(ip)) {
//                ip = request.getRemoteAddr();
//            }
//        } else if (ip.length() > 15) {
//            String[] ips = ip.split(",");
//            for (int index = 0; index < ips.length; index++) {
//                String strIp = (String) ips[index];
//                if (!("unknown".equalsIgnoreCase(strIp))) {
//                    ip = strIp;
//                    break;
//                }
//            }
//        }
//        return ip;
//    }

    // 搜索页面
    @RequestMapping(value = "/list",method = RequestMethod.GET)
    public ResponseEntity<Map<String,Object>> getMessageById(@RequestParam("key") String key
            ,@RequestParam("err_b")String err_b
            ,@RequestParam("err_e") String err_e
            ,@RequestParam("use_b") String use_b
            ,@RequestParam("use_e") String use_e
            ,@RequestParam("volt") String volt
            ,@RequestParam("err_device") String err_device
            ,@RequestParam("err_level") String err_level){
        return reportService.getErrorReport(key,err_b,err_e,use_b,use_e,volt,err_device,err_level);
    }

//    @RequestMapping(value = "/list/{id}",method = RequestMethod.GET)
//    public ResponseEntity<Map<String,Object>> getFullTextById(@PathVariable("id") String id){
//        return reportService.getFullReport(id);
//    }

    @RequestMapping(value = "/detail/{id}",method = RequestMethod.GET)
    public String getFullTextHtmlById(@PathVariable String id, Model model){
        //reportService.getFullReport(id);
        ErrorReport report = reportService.getFullReport(id);
        model.addAttribute("baseurl", baseurl);
        model.addAttribute("report", report);
        String src = "/docs/"+report.getReport_name()+".pdf";
        model.addAttribute("pdf_src",src.replace("#","-"));
        return "report";
    }
    @RequestMapping(value = "/source_example/{id}",method = RequestMethod.GET)
    public String getExample(@PathVariable String id, Model model){
        System.out.println("web123");
        model.addAttribute("baseurl", baseurl);
        ErrorReport report = reportService.getFullReport(id);
        model.addAttribute("query_report", report);
        return "example";
    }


    @RequestMapping(value = "/example/list",method = RequestMethod.POST)
    public ResponseEntity<Map<String,Object>> getExampleByReportName(@RequestBody JSONObject jsonObject){
        System.out.println("wh"+"");
        List<String> a = (List<String>) jsonObject.get("report_name");
        System.out.println(a.get(0));
        return reportService.getExampleBy_report_name(a);
    }
    @RequestMapping(value = "/export/{id}",method = RequestMethod.GET)
    public String getExportHtmlById(@PathVariable String id, Model model){
        //reportService.getFullReport(id);
        ErrorReport report = reportService.getFullReport(id);
//        model.addAttribute("baseurl", baseurl);
        model.addAttribute("report", report);
        String src = "/docs/"+report.getReport_name()+".pdf";
        model.addAttribute("pdf_src",src.replace("#","-"));
        return "export";
    }

    @RequestMapping(value = "/getPics",method = RequestMethod.GET)
    public ResponseEntity<List<ErrorReportPics>> getById(@Param("id") String id){
        //reportService.getFullReport(id);
        return reportService.getReportPic(id);
    }

    @RequestMapping(value = "/main",method = RequestMethod.GET)
    public String getMain( Model model ){
        model.addAttribute("baseurl", baseurl);
        return "main";
    }

    @RequestMapping(value = "/search",method = RequestMethod.GET)
    public String getSearch( Model model ){
        model.addAttribute("baseurl", baseurl);
        return "search";
    }

    @RequestMapping(value = "/advice",method = RequestMethod.GET)
    public String getAdvice( Model model ){
        model.addAttribute("baseurl", baseurl);
        return "advice";
    }

    @RequestMapping(value = "/updateReport",method = RequestMethod.POST)
    @ResponseBody
    public String updateReport(@RequestBody ErrorReport report){

        return reportService.updateErrorReportByID(report);
    }
    //柱状图表的季度
    @RequestMapping(value = "/countSeason",method = RequestMethod.GET)
    public ResponseEntity getCountSeason(@RequestParam("keyword") String kw) {

        return reportService.countBySeason(kw);
    }
    //柱状图表的时间
    @RequestMapping(value = "/countTime",method = RequestMethod.GET)
    public ResponseEntity getCountTime(@RequestParam("keyword") String kw) {

        return reportService.countByErrHourSplit(kw);
    }

    @RequestMapping(value = "/countDevType",method = RequestMethod.GET)
    public ResponseEntity getCountDevType(@RequestParam("keyword") String kw) {

        return reportService.countByDevType(kw);
    }
    //柱状图表的电压
    @RequestMapping(value = "/countVolt",method = RequestMethod.GET)
    public ResponseEntity getCountVolt(@RequestParam("keyword") String kw) {
        return reportService.countBykV(kw);
    }

    @RequestMapping(value = "/getPDF", method = RequestMethod.GET)
    public ResponseEntity getPDF(@RequestParam("id") String id) {
        return reportService.html2pdf(id);
    }
}
