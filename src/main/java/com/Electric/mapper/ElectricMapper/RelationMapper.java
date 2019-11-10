package com.Electric.mapper.ElectricMapper;

import com.Electric.model.Node;
import com.Electric.model.Link;
import com.Electric.model.WordCloud;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Result;
import org.apache.ibatis.annotations.Results;
import org.apache.ibatis.annotations.Select;


import java.util.List;

public interface RelationMapper {

    @Select("SELECT requirement " +
            ",count(*) as value " +
            ",req_type as category " +
            "FROM relation " +
            "GROUP BY requirement,req_type;" )
    @Results(id="nodeMap", value={
            @Result(property = "name", column = "requirement"),
            @Result(property = "value", column = "value"),
            @Result(property = "category", column = "category")
    })
    List<Node> selectNodes();

    @Select("SELECT * " +
            "FROM relation ;" )
    @Results(value={
            @Result(property = "source", column = "requirement"),
            @Result(property = "target", column = "conclusion"),
            @Result(property = "support", column = "support"),
            @Result(property = "confidence", column = "confidence")
    })
    List<Link> selectLinks();

    @Select("SELECT count(distinct req_type) as category " +
            "FROM relation ;" )
    int selectCategory();


    @Select("<script>" +
            "SELECT * " +
            "FROM relation " +
            "<if test=\"kw!=null and kw!=''\">" +
            "WHERE requirement = #{kw} " +
            "OR conclusion = #{kw}" +
            "</if>" +
            "ORDER BY confidence desc" +
            "</script>" )
    @Results(value={
            @Result(property = "source", column = "requirement"),
            @Result(property = "target", column = "conclusion"),
            @Result(property = "support", column = "support"),
            @Result(property = "confidence", column = "confidence")
    })
    List<Link> selectLinksByRequirement(@Param("kw") String key);

    @Select("SELECT * " +
            "FROM word_cloud " +
            "where keyi in " +
            "(select distinct requirement from relation) " +
            "or keyi = 'global';" )
    @Results(

            id="cloudMap", value={
            @Result(property = "key", column = "keyi"),
            @Result(property = "value", column = "value")
    })
    List<WordCloud> selectWordCloud();




}
