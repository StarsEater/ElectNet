package com.Electric.mapper.ElectricMapper;
import com.Electric.model.Decision;
import com.Electric.model.SelectParams;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import java.util.List;

public interface SelectParamsMapper {

    @Select("SELECT * " +
            "FROM select_param " +
            "ORDER BY classes;")
    List<SelectParams> selectParams();

    @Select("<script>" +
            "SELECT * " +
            "FROM decision " +
            "WHERE kind = #{kind} " +
            "AND param = #{param} " +
            "<if test='desc!=null and desc!=\"\"'>" +
            "AND description = #{desc}" +
            "</if>" +
            "ORDER BY kind;" +
            "</script>")
    List<Decision> selectDecisions(@Param("kind")String c, @Param("param") String p, @Param("desc") String desc);
}
