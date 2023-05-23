package silver.vision.silver;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import silver.vision.silver.utils.Utils;

@SpringBootApplication
public class SilverApplication {

    public static void main(String[] args) {
        SpringApplication.run(SilverApplication.class, args);
        Utils.serializer();
    }

}
