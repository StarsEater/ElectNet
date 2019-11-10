package com.Electric.service;


import com.Electric.mapper.ElectricMapper.RelationMapper;
import com.Electric.model.Link;
import com.Electric.model.Node;
import com.Electric.model.WordCloud;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class RelationService {
    @Autowired
    private RelationMapper mapper;

    public RelationService(RelationMapper relationMapper) {
        this.mapper = relationMapper;
    }

    public ResponseEntity<Map<String,Object>> getRelationMap(){
        HashMap<String,Object> res = new HashMap<>();
        List<Node> nodes = null;
        List<Link> links = null;
        List<WordCloud> wordClouds = null;
        int category = mapper.selectCategory();

        nodes = mapper.selectNodes();
        links = mapper.selectLinks();
        wordClouds = mapper.selectWordCloud();
        HashMap<String,String> worddict = new HashMap<>();
        for(WordCloud w :wordClouds){
            if(w.getKey()!=null)
            worddict.put(w.getKey(),w.getMap());
        }

        res.put("nodes",nodes);
        res.put("links",links);
        res.put("words",worddict);
        res.put("category",category);
        res.put("succeed",1);
        res.put("message","success");
        return new ResponseEntity<>(res,HttpStatus.OK);

    }

    public ResponseEntity<Map<String,Object>> getRelationTable(String key){
        HashMap<String,Object> res = new HashMap<>();
        List<Link> links = null;
        links = mapper.selectLinksByRequirement(key);
        if(links.size()>10)
            links = links.subList(0,10);
        res.put("links",links);
        res.put("succeed",1);
        res.put("message","success");
        return new ResponseEntity<>(res,HttpStatus.OK);

    }
}
