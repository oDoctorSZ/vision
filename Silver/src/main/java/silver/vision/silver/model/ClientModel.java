package silver.vision.silver.model;

public class ClientModel {

    private String token;
    private String nome;
    private String sobrenome;
    private String email;
    private String password;

    public ClientModel(String token, String nome, String sobrenome, String email, String password) {
        this.token = token;
        this.nome = nome;
        this.sobrenome = sobrenome;
        this.email = email;
        this.password = password;
    }

    public String getToken() {
        return token;
    }

    public void setToken(String token) {
        this.token = token;
    }

    public String getFirstName() {
        return nome;
    }

    public void setFirstName(String nome) {
        this.nome = nome;
    }

    public String getSurName() {
        return sobrenome;
    }

    public void setSurName(String sobrenome) {
        this.sobrenome = sobrenome;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }
}
