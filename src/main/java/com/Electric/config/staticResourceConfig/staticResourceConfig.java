package com.Electric.config.staticResourceConfig;

import com.Electric.Utils.UserUtil;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurerAdapter;

/**
 * Created by Muki on 2018/1/12
 */
@Configuration
public class staticResourceConfig extends WebMvcConfigurerAdapter {
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry resourceHandlerRegistry) {
        resourceHandlerRegistry.addResourceHandler("static/**").addResourceLocations("classpath:/static/");
        String picsImgPath = UserUtil.getUpperUFolderPath("/pics/");
        resourceHandlerRegistry.addResourceHandler("pics/**").addResourceLocations("file:" + picsImgPath);
        String docHtmlPath = UserUtil.getUpperUFolderPath("/docs/");
        resourceHandlerRegistry.addResourceHandler("docs/**").addResourceLocations("file:" + docHtmlPath);
        super.addResourceHandlers(resourceHandlerRegistry);
    }
}