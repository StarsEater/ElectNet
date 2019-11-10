package com.Electric.config.filterRegisterConfig;

import com.Electric.filter.RequestAndResponseFilter;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Created by Muki on 2017/11/21
 */

@Configuration
public class FilterRegistration {
    @Bean
    public FilterRegistrationBean indexFilterRegistration() {
        FilterRegistrationBean registration = new FilterRegistrationBean(new RequestAndResponseFilter());
        registration.setOrder(0);
        return registration;
    }
}