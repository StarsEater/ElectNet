package com.Electric;
/**
 * Created by Muki on 2017/10/15
 */

import org.apache.catalina.connector.Connector;
import org.mybatis.spring.annotation.MapperScan;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.embedded.EmbeddedServletContainerFactory;
import org.springframework.boot.context.embedded.tomcat.TomcatEmbeddedServletContainerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@SpringBootApplication
@EnableAsync
@EnableTransactionManagement
@MapperScan("com.Electric.mapper")
public class ElectricApplication {
    public static void main(String[] args) {
        Logger logger = LoggerFactory.getLogger(ElectricApplication.class);
        logger.debug("start application");
        System.setProperty("spring.devtools.restart.enabled","false");
        SpringApplication.run(ElectricApplication.class, args);
    }

    @Value("${http.port}")
    private Integer port;

    // 这是spring boot 2.0.X版本的 添加这个，上一个就不用添加了
    @Bean
    public EmbeddedServletContainerFactory servletContainer() {
        TomcatEmbeddedServletContainerFactory tomcat = new TomcatEmbeddedServletContainerFactory();
        tomcat.addAdditionalTomcatConnectors(createStandardConnector()); // 添加http
        return tomcat;
    }

    // 配置http
    private Connector createStandardConnector() {
        Connector connector = new Connector("org.apache.coyote.http11.Http11NioProtocol");
        connector.setPort(port);
        return connector;
    }
}