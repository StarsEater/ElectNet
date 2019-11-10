package com.Electric.service;

import com.Electric.mapper.ElectricMapper.MendMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MendService {
    @Autowired
    private MendMapper mendMapper;
    public MendService(MendMapper mendMapper) {
        this.mendMapper = mendMapper;
    }
    public  String getAdviceFromPy(String content) throws IOException, InterruptedException {
        String ans = "";
        File fp=new File("D:\\study\\EN\\Electric-Err1\\Algorithm\\multi_label\\t.txt");
        String txtPath = "D:\\study\\EN\\Electric-Err1\\Algorithm\\multi_label\\s.txt";
        PrintWriter pfp= new PrintWriter(fp);
        pfp.print(content);
        pfp.close();
        String exe = "python";
        String command = "D:\\study\\EN\\Electric-Err1\\Algorithm\\multi_label\\advice.py";
        String[] cmdArr = new String[] { exe, command};
        Process process = Runtime.getRuntime().exec(cmdArr);

        Thread.sleep(20000);
        InputStream is = new FileInputStream(txtPath);
        int iAvail = is.available();
        byte[] bytes = new byte[iAvail];
        is.read(bytes);
        is.close();

        ans = new String(bytes);
        int result = process.waitFor();
        System.out.println("执行结果:" + result+ ans);

        return ans;
    }
    public ResponseEntity<Map<String,Object>> getAiAdvice(String text) throws IOException, InterruptedException {
        String advice = getAdviceFromPy(text);
        HashMap<String,Object> res = new HashMap<>();
        res.put("advice",advice);
        res.put("succeed",1);
        res.put("message","success");
        return new ResponseEntity<>(res, HttpStatus.OK);
    }

//    public static void main(String[] args) throws IOException, InterruptedException {
//       String s = "1.案例经过\n" +
//                "河南省电力公司某220千伏变电站#3主变2010年12月25日投运。主变型号：SSZ10-1800000/220，2010年6月出厂。变压器安装试验数据全部符合规程要求。现场运行中发现铁心接地电流超标，后经放油检查铁芯引出线与变压器外壳间金属短接，经处理后变压器投入正常运行。\n" +
//                "2.检测分析方法\n" +
//                "#3主变在投运后近1年后，运行人员发现其铁心接地电流超过0.1A的注意值。2012年3月11日，在进行#3主变运行一年时的交接试验时，变压器铁芯与地之间的绝缘测试结果为0Ω，现场使用36µf的电容器、80A的电焊机多次对铁芯进行电流冲击。冲击后，测试铁芯对地绝缘依然为0Ω，铁芯多点接地故障无法通过电流冲击进行消除。\n" +
//                "为进一步判断铁芯多点接地故障性质，将#3主变恢复运行，测得数据如下： #3主变空载时，铁芯接地电流114 mA；负荷2.4万千伏安时，铁芯接地电流1.3A；负荷3.6万千伏安时，铁芯接地电流2.7A。#3主变绝缘油化验结果显示：总烃1.66µL/L，乙烯0.16µL/L，属合格范围之内。\n" +
//                "根据以上情况，判断#3主变铁芯接地性质为：金属接地，故障接地点与正常接地点之间无环流发生，不会引起变压器线圈发热等故障。\n" +
//                "5月22日，现场将变压器绝缘油放至顶部下方10cm处，由铁芯接地套管处进行排查。拆除铁芯引出线套管后，将铁芯接地引出线进行了拉伸、摆动，测得铁芯对地绝缘为20MΩ。将4层绝缘纸板塞入铁芯引出线与外壳之间的缝隙后，测得铁芯对地绝缘为1000MΩ，判定故障为铁芯引出线与变压器外壳金属短接，如图1-1所示。\n" +
//                "因变压器没有进行吊罩工作，受检修条件限制，现场加工绝缘纸板后，将绝缘纸板固定在铁芯引出线固定木架上，有效的隔离了铁芯引出线与变压器外壳的金属连接。\n" +
//                "变压器注油后，测得变压器各项数据均在规程要求范围，本次故障处理完毕。";
//       String ans = getAdviceFromPy(s);
//       System.out.println(ans);
//    }
}
