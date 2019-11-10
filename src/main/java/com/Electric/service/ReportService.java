package com.Electric.service;

import com.Electric.Utils.UserUtil;
import com.Electric.model.ErrorReportCount;
import com.Electric.model.ErrorReportPics;
import com.Electric.mapper.ElectricMapper.ErrorReportMapper;
import com.Electric.model.ErrorReport;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class ReportService {
    @Autowired
    private ErrorReportMapper mapper;

    public ReportService(ErrorReportMapper errorReportMapper) {
        this.mapper = errorReportMapper;
    }

    // 检索满足条件的报告
    public ResponseEntity<Map<String,Object>> getErrorReport(String key,String err_b,String err_e,String use_b,String use_e,String volt,String err_device,String err_level){
        HashMap<String,Object> res = new HashMap<>();
        List<ErrorReport> reports = null;
        String[] err_bs = err_b.split("-");
        String[] err_es = err_e.split("-");
        String[] use_bs = use_b.split("-");
        String[] use_es = use_e.split("-");
        if(volt.equals("所有"))
            volt = null;
        else if(volt.equals("其他"))
            volt = "";
        if(err_device.equals("所有"))
            err_device = null;
        if(err_level.equals("所有"))
            err_level = null;
        else if(err_level.equals("其他"))
            err_level = "";

        reports = mapper.selectErrorReportByReport_key(key,err_bs[0],err_bs[1],err_es[0],err_es[1],use_bs[0],use_bs[1],use_es[0],use_es[1],volt,err_device,err_level);
        res.put("reports",reports);
        res.put("succeed",1);
        res.put("message","success");
        return new ResponseEntity<>(res,HttpStatus.OK);
    }

    // 故障报告详情
    public ErrorReport getFullReport(String id){
        List<ErrorReport> reports = null;
        reports = mapper.selectErrorReportByID(id);
        return reports.get(0);
    }

    public ResponseEntity<List<ErrorReportPics>> getReportPic(String id){
        return new ResponseEntity<>(mapper.getErrorReportPicsByID(id), HttpStatus.ACCEPTED);
    }

    // 故障报告数量随年份变化
    public ResponseEntity<List<ErrorReportCount>> countByErrYear(String kw){
        return new ResponseEntity<>(mapper.countErrorReoprtByKeyWord("err_time_y",kw), HttpStatus.ACCEPTED);
    }

    // 电压等级
    public ResponseEntity<List<ErrorReportCount>> countBykV(String kw){
        return new ResponseEntity<>(mapper.countVoltByKeyWord(kw), HttpStatus.ACCEPTED);
    }

    // 故障设备类型
    public ResponseEntity<List<ErrorReportCount>> countByDevType(String kw){
        return new ResponseEntity<>(mapper.countDevTypeByKeyWord(kw), HttpStatus.ACCEPTED);
    }

    // 根据季节统计
    public ResponseEntity<List<ErrorReportCount>> countBySeason(String kw){
        return new ResponseEntity<>(mapper.countSeasonByKeyWord(kw), HttpStatus.ACCEPTED);
    }

    // 根据故障时间（分组）
    public ResponseEntity<List<ErrorReportCount>> countByErrHourSplit(String kw){
        List<ErrorReportCount> ret = mapper.countTimeByKeyWord(kw);
        if(ret.size() <= 1){
            ret = mapper.countDevTypeByKeyWord(kw);
        }
        return new ResponseEntity<>(ret, HttpStatus.ACCEPTED);
    }

    public String updateErrorReportByID(ErrorReport report){
        if(report.getUse_time()!=null) {
            try {
                String[] errs = report.getUse_time().split("-");
                if (errs.length == 1) {
                    report.setErr_time_y(Integer.parseInt(errs[0]));
                    report.setErr_time_m(-1);
                    report.setErr_time_d(-1);
                } else if (errs.length == 2) {
                    report.setErr_time_y(Integer.parseInt(errs[0]));
                    report.setErr_time_m(Integer.parseInt(errs[1]));
                    report.setErr_time_d(-1);
                } else if (errs.length == 3) {
                    report.setErr_time_y(Integer.parseInt(errs[0]));
                    report.setErr_time_m(Integer.parseInt(errs[1]));
                    report.setErr_time_d(Integer.parseInt(errs[2]));
                } else return "日期格式不正确";
                String[] uses = report.getUse_time().split("-");
                if (uses.length == 1) {
                    report.setUse_time_y(Integer.parseInt(uses[0]));
                    report.setUse_time_m(-1);
                    report.setUse_time_d(-1);
                } else if (uses.length == 2) {
                    report.setUse_time_y(Integer.parseInt(uses[0]));
                    report.setUse_time_m(Integer.parseInt(uses[1]));
                    report.setUse_time_d(-1);
                } else if (uses.length == 3) {
                    report.setUse_time_y(Integer.parseInt(uses[0]));
                    report.setUse_time_m(Integer.parseInt(uses[1]));
                    report.setUse_time_d(Integer.parseInt(uses[2]));
                } else return "日期格式不正确";
            }
            catch (RuntimeException e){
                return "日期格式不正确";
            }
            mapper.updateErrorReportByID(report);
        }
        return "update";
    }

    public ResponseEntity<Map<String, String>> html2pdf(String id){
        String url = "http://localhost:8085/report/export/" + id;
        StringBuilder cmd = new StringBuilder();
        cmd.append("wkhtmltopdf ");
        cmd.append("--stop-slow-scripts ");
        cmd.append(url);
        cmd.append(" ");
        cmd.append(UserUtil.getUpperUFolderPath("/pics/"));
        cmd.append(id);
        cmd.append(".pdf");
        HashMap<String, String> ret = new HashMap<>();
        ret.put("succeed", "1");
        ret.put("url", "/pics/" + id + ".pdf");
        System.out.println(cmd.toString());
        try{
            Process proc = Runtime.getRuntime().exec(cmd.toString());
            proc.waitFor();
        }catch(Exception e){
            e.printStackTrace();
        }
        return new ResponseEntity<>(ret, HttpStatus.ACCEPTED);
    }
}
