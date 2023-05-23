package silver.vision.silver.controllers;

import org.springframework.web.bind.annotation.*;
import silver.vision.silver.model.ClientModel;
import silver.vision.silver.utils.Utils;

import java.util.Random;
import java.util.UUID;


@RestController
@RequestMapping("/api/v1")
public class LoginController {

    @CrossOrigin(origins = "http://localhost:5173", maxAge = 3600)
    @PostMapping(path = "register")
    public @ResponseBody ClientModel registerClient(@RequestParam String nome, @RequestParam String sobrenome, @RequestParam String email, @RequestParam String password) {
        ClientModel clientModel = new ClientModel(UUID.randomUUID().toString(), nome, sobrenome, email, password);
        Utils.cList.add(clientModel);
        return clientModel;
    }

    @GetMapping(path = "login")
    public ClientModel loginClient(@RequestParam String email, @RequestParam String password) {
        for (ClientModel client : Utils.cList) {
            if (client.getEmail().equals(email) && client.getPassword().equals(password)) {
                return client;
            }
        }
        return null;

    }

}
