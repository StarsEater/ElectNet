package com.Electric.service;

import com.Electric.mapper.ElectricMapper.SelectParamsMapper;
import com.Electric.model.Decision;
import com.Electric.model.SelectParams;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class ParamsService {
    @Autowired
    private SelectParamsMapper mapper;


    public ParamsService(SelectParamsMapper selectParamsMapper) {
        this.mapper = selectParamsMapper;
    }

    public ResponseEntity<Map<String,Object>> getSelectParams() {
        HashMap<String,Object> res = new HashMap<>();
        List<SelectParams> sets = null;
        sets = mapper.selectParams();

        HashMap<String, HashMap<String, List<String>>> classes = new HashMap<>();
        // classes -- <classes,(<param,des>,<para,des>)
        // 同一类别的参数和描述放在了classes Map 里 返回
        for(SelectParams p : sets) {
            if(!classes.containsKey(p.getClasses())) classes.put(p.getClasses(), new HashMap<>());
            if(!classes.get(p.getClasses()).containsKey(p.getParams())) classes.get(p.getClasses()).put(p.getParams(), new ArrayList<>());
            classes.get(p.getClasses()).get(p.getParams()).add(p.getDescription());
        }

        res.put("dict",classes);
        res.put("succeed",1);
        res.put("message","success");
        return new ResponseEntity<>(res,HttpStatus.OK);
    }

    public ResponseEntity<Map<String,Object>> getDecisions(String kind,String param, String desc) {
        HashMap<String,Object> res = new HashMap<>();
        List<Decision> decisions = null;
        // 选择具体种类、参数、描述的数据返回
        decisions = mapper.selectDecisions(kind,param, desc);
        res.put("decisions",decisions);
        res.put("succeed",1);
        res.put("message","success");
        return new ResponseEntity<>(res,HttpStatus.OK);
    }
}
