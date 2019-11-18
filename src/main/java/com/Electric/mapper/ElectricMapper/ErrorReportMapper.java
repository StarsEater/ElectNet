package com.Electric.mapper.ElectricMapper;

import com.Electric.model.ErrorReport;

import com.Electric.model.ErrorReportCount;
import com.Electric.model.ErrorReportPics;
import org.apache.ibatis.annotations.*;


import java.util.List;

public interface ErrorReportMapper {

    @Select("<script>" +
            "SELECT report_name,err_name,err_how,err_time,err_level,volt,device_type,type_code,tower_code,use_time,substation,report_id," +
            "concat_ws('-',err_time_y,err_time_m) as err_search , " +
            "concat_ws('-',use_time_y,use_time_m) as use_search  " +
            "FROM electric_err " +
            "WHERE " +
            "((err_time_y = #{errby} AND err_time_y = #{errey} AND ((err_time_m &lt;= #{errbm} AND err_time_m &lt;= #{errem}) OR err_time_m = -1))" +
            "OR (err_time_y = #{errby} AND err_time_y &lt; #{errey} AND (err_time_m &gt;= #{errbm}  OR err_time_m = -1)) " +
            "OR (err_time_y &gt; #{errby} AND err_time_y = #{errey} AND err_time_m &lt;= #{errbm}) " +
            "OR (err_time_y &gt; #{errby} AND err_time_y &lt; #{errey} )) " +
            "AND ((use_time_y = #{useby} AND use_time_y = #{useey} AND ((use_time_m &lt;= #{usebm} AND use_time_m &lt;= #{useem}) OR use_time_m = -1))" +
            "OR (use_time_y = #{useby} AND use_time_y &lt; #{useey} AND (use_time_m &gt;= #{usebm}  OR use_time_m = -1)) " +
            "OR (use_time_y &gt; #{useby} AND use_time_y = #{useey} AND use_time_m &lt;= #{usebm}) " +
            "OR (use_time_y &gt; #{useby} AND use_time_y &lt; #{useey} )) " +
            "<if test=\"volt!=null and volt!=''\">" +
            "AND volt = #{volt} " +
            "</if>" +
            "<if test=\"volt==''\">" +
            "AND volt NOT IN ('110kV', '220kV') " +
            "</if>" +
            "<if test=\"key!=null and key!=''\">" +
            "AND report_name like CONCAT('%',#{key},'%') " +
            "</if>" +
            "<if test=\"err_device!=null and err_device!=''\">" +
            "AND err_device_type = #{err_device} " +
            "</if>" +
            "<if test=\"err_level!=null and err_level!=''\">" +
            "AND err_level = #{err_level} " +
            "</if>" +
            "<if test=\"err_level==''\">" +
            "AND err_level is null " +
            "</if>" +
            "</script>")
    List<ErrorReport> selectErrorReportByReport_key(
            @Param("key") String key,
            @Param("errby") String errby,
            @Param("errbm") String errbm,
            @Param("errey") String errey,
            @Param("errem") String errem,
            @Param("useby") String useby,
            @Param("usebm") String usebm,
            @Param("useey") String useey,
            @Param("useem") String useem,
            @Param("volt") String volt,
            @Param("err_device") String err_device,
            @Param("err_level") String err_levelgit
    );

    @Select("SELECT * " +
            "FROM electric_err " +
            "WHERE report_id = #{id};")
    List<ErrorReport> selectErrorReportByID(@Param("id") String id);
    @Select("SELECT report_name,err_name,err_how,err_time,err_level,volt,device_type,type_code,tower_code,use_time,substation,report_id," +
            "concat_ws('-',err_time_y,err_time_m) as err_search , " +
            "concat_ws('-',use_time_y,use_time_m) as use_search  " +
            "FROM electric_err " +
            "WHERE report_name in (${names});")
    List<ErrorReport> selectExampleByName(@Param("names") String names);

    // 对某个字段分组计数
    @Select("SELECT err_season as con, count(*) as num " +
            "FROM electric_err " +
            "WHERE err_season is not null " +
            "AND rule_keywords LIKE CONCAT('%',#{keyword},'%') " +
            "GROUP BY err_season ORDER BY err_season ASC")
    @Results(value={
            @Result(property = "condition", column = "con"),
            @Result(property = "num", column = "num")
    })
    List<ErrorReportCount> countSeasonByKeyWord(@Param("keyword") String keyword);

    // 对某个字段分组计数
    @Select("SELECT err_hour_split as con, count(*) as num " +
            "FROM electric_err " +
            "WHERE err_hour_split is not null " +
            "AND rule_keywords LIKE CONCAT('%',#{keyword},'%') " +
            "GROUP BY err_hour_split ORDER BY err_hour_split ASC")
    @Results(value={
            @Result(property = "condition", column = "con"),
            @Result(property = "num", column = "num")
    })
    List<ErrorReportCount> countTimeByKeyWord(@Param("keyword") String keyword);

    // 对某个字段分组计数
    @Select("SELECT err_device_type as con, count(*) as num " +
            "FROM electric_err " +
            "WHERE err_device_type is not null " +
            "AND rule_keywords LIKE CONCAT('%',#{keyword},'%') " +
            "GROUP BY err_device_type ORDER BY err_device_type ASC")
    @Results(value={
            @Result(property = "condition", column = "con"),
            @Result(property = "num", column = "num")
    })
    List<ErrorReportCount> countDevTypeByKeyWord(@Param("keyword") String keyword);
    
    @Select("SELECT volt as con, count(*) as num " +
            "FROM electric_err " +
            "WHERE volt is not null " +
            "AND rule_keywords LIKE CONCAT('%', #{keyword}, '%') " +
            "GROUP BY volt ORDER BY volt ASC;")
    @Results(value={
            @Result(property = "condition", column = "con"),
            @Result(property = "num", column = "num")
    })
    List<ErrorReportCount> countVoltByKeyWord(@Param("keyword") String keyword);

    @Select("SELECT #{condition} as con, count(1) as num " +
            "FROM electric_err " +
            "WHERE  rule_keywords " +
            "LIKE CONCAT('%',#{keyword},'%') " +
            "GROUP BY #{condition} ORDER BY #{condition} ASC")
    @Results(value={
            @Result(property = "condition", column = "con"),
            @Result(property = "num", column = "num")
    })
    List<ErrorReportCount> countErrorReoprtByKeyWord(@Param("condition") String condition, @Param("keyword") String keyword);


    // 读取图片列表
    @Select("SELECT pic_path FROM electric_err_pics WHERE report_id = #{id};")
    List<ErrorReportPics> getErrorReportPicsByID(@Param("id") String id);


//    @Select("<script>" +
//            "SELECT distinct " +
//            "<if test=\"key=='device_type'\">" +
//            "device_type " +
//            "</if>" +
//            "<if test=\"key=='volt'\">" +
//            "vol " +
//            "</if>" +
//            "<if test=\"key=='device_type'\">" +
//            "device_type " +
//            "</if>" +
//            "FROM eletric_err " +
//            "</script>")
//    List<String> getErrorReportDistinct(@Param("key") String key);

    @Update("Update electric_err SET " +
            "advice = #{advice}" +
            ",analysis = #{analysis}" +
            ",conclusion = #{conclusion}" +
            ",description = #{description}" +
            ",device_type = #{device_type}" +
            ",dispose = #{dispose}" +
            ",err_how = #{err_how}" +
            ",err_level = #{err_level}" +
            ",err_name = #{err_name}" +
            ",err_time = #{err_time}" +
            ",extra_time = #{extra_time}" +
            ",factory = #{factory}" +
            ",humidity = #{humidity}" +
            ",situation = #{situation}" +
            ",substation = #{substation}" +
            ",temperature = #{temperature}" +
            ",tower_code = #{tower_code}" +
            ",type_code = #{type_code}" +
            ",use_time = #{use_time}" +
            ",volt = #{volt}" +
            "WHERE report_id =#{report_id}")
    void updateErrorReportByID(ErrorReport report);

}
