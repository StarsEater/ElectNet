package com.Electric.Utils;

/**
 * Created by BigHeart on 2017/11/05
 */
public class UserUtil {
//    private final static boolean IS_WINDOWS = false;
    private final static boolean IS_WINDOWS = true;


    /**
     * 获取当前运行路径的上一级文件夹路径，一般在上一级文件夹路径中存储可更新的资源
     *
     * @param sourceFolder 上一级文件夹路径中可更新的资源的路径，如头像资源，则传入：/avator/
     * @return 当前运行路径的上一级文件夹路径
     */
    public static String getUpperUFolderPath(String sourceFolder) {

        String CRT_PATH = System.getProperty("user.dir");
        if (IS_WINDOWS) {
            CRT_PATH = CRT_PATH.substring(0, CRT_PATH.lastIndexOf('\\'));
            sourceFolder = sourceFolder.replace('/', '\\');
        } else {
            CRT_PATH = CRT_PATH.substring(0, CRT_PATH.lastIndexOf('/'));
        }
        return CRT_PATH + sourceFolder;
    }
}
